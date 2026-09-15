"""
Graduated driver alerting with offline text-to-speech.

Design rules
------------
1. NEVER block the inference loop.
   The original inference.py called winsound.Beep(2000, 100) inline. Beep is
   synchronous: it froze the loop for 100 ms on every frame while the alarm
   was active, dropping the system to ~10 FPS exactly when it had to be
   responsive. Here every sound is dispatched to a worker thread through a
   queue; the loop calls notify() and returns immediately.

2. Escalate, do not startle.
   A sudden loud alarm makes a drowsy driver jerk the wheel. Mercedes
   Attention Assist and Volvo Driver Alert both use a soft chime plus a
   "take a break" suggestion, not a siren. Severity rises only if the
   condition persists.

3. Do not repeat.
   Without a cooldown the system would speak on every frame. Each severity
   has its own minimum interval, and a message is never repeated verbatim
   back to back.

4. Degrade gracefully.
   If pyttsx3 or the audio device is missing (headless server, container,
   Raspberry Pi without a DAC), the module falls back to console output
   and keeps running. Audio is a feature, not a dependency.

SAFETY NOTE
-----------
This is a demonstration, not a certified driver monitoring system. No
alerting strategy replaces stopping to rest: the only effective
countermeasures to drowsiness are caffeine and a short nap. Messages here
push the driver to stop, never to keep going.

Install (optional)
------------------
    pip install pyttsx3
"""

import math
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from drowsiness_state import AlertLevel


class Severity(Enum):
    NONE = 0
    NOTICE = 1      # quiet visual cue, no sound
    WARNING = 2     # soft chime + short spoken hint
    ALARM = 3       # firm chime + spoken instruction to pull over


@dataclass
class AlertConfig:
    # minimum seconds between two alerts of the same severity
    cooldown_notice: float = 60.0
    cooldown_warning: float = 45.0
    cooldown_alarm: float = 20.0

    # escalate to ALARM if DROWSY persists this long
    escalate_after_s: float = 30.0

    enable_voice: bool = True
    enable_chime: bool = True
    voice_rate: int = 165          # words per minute; slower than default
    voice_volume: float = 0.9


MESSAGES = {
    Severity.WARNING: [
        "Your attention seems to be dropping. Consider a break soon.",
        "Signs of fatigue detected. A short stop would help.",
    ],
    Severity.ALARM: [
        "Drowsiness detected. Please pull over at the next safe place.",
        "You are falling asleep. Stop the car and rest.",
    ],
}


# ----------------------------------------------------------------------
def _write_chime(path: str, freq_hz: float, ms: int, volume: float) -> None:
    """Generate a soft sine chime with fade in/out.

    Written by hand rather than shipping a .wav: no binary asset in the
    repository, and the tone stays tunable. The fade matters — a raw sine
    that starts at full amplitude clicks, which is exactly the startling
    sound we are trying to avoid.
    """
    rate = 22050
    n = int(rate * ms / 1000)
    fade = int(rate * 0.04)
    frames = bytearray()
    for i in range(n):
        env = 1.0
        if i < fade:
            env = i / fade
        elif i > n - fade:
            env = (n - i) / fade
        s = math.sin(2 * math.pi * freq_hz * i / rate) * volume * env
        frames += struct.pack("<h", int(s * 32767))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def _play_wav(path: str) -> None:
    """Best-effort playback. Silence is acceptable; a crash is not."""
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        elif sys.platform == "darwin":
            subprocess.run(["afplay", path], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["aplay", "-q", path], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ----------------------------------------------------------------------
class AlertManager:
    """
        alerts = AlertManager()
        ...
        alerts.notify(t_video, state)     # returns immediately
        ...
        alerts.close()
    """

    def __init__(self, cfg: Optional[AlertConfig] = None):
        self.cfg = cfg or AlertConfig()
        self._q: "queue.Queue" = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._last_fired = {s: -1e9 for s in Severity}
        self._last_text = ""
        self._drowsy_since: Optional[float] = None
        self._msg_idx = 0
        self.history = []               # (t, severity, text) for the report

        self._tmp = tempfile.mkdtemp(prefix="drowsy_alerts_")
        self._chimes = {}
        if self.cfg.enable_chime:
            try:
                for sev, (f, ms, vol) in {
                    Severity.WARNING: (660.0, 260, 0.25),
                    Severity.ALARM: (880.0, 420, 0.45),
                }.items():
                    p = os.path.join(self._tmp, f"{sev.name.lower()}.wav")
                    _write_chime(p, f, ms, vol)
                    self._chimes[sev] = p
            except Exception as e:
                print(f"[alerts] chime generation failed ({type(e).__name__}), "
                      f"continuing without sound")
                self._chimes = {}

        self._tts = None
        if self.cfg.enable_voice:
            try:
                import pyttsx3
                self._tts = pyttsx3.init()
                self._tts.setProperty("rate", self.cfg.voice_rate)
                self._tts.setProperty("volume", self.cfg.voice_volume)
            except Exception as e:
                print(f"[alerts] TTS unavailable ({type(e).__name__}), "
                      f"messages will be printed instead")
                self._tts = None

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    def notify(self, t: float, state) -> Optional[Severity]:
        """Call once per frame. Returns the severity fired, or None.

        t is the clip/video timestamp, so cooldowns follow the driver's
        timeline rather than the machine's processing speed.
        """
        sev = self._severity(t, state)
        if sev is Severity.NONE:
            return None

        cooldown = {
            Severity.NOTICE: self.cfg.cooldown_notice,
            Severity.WARNING: self.cfg.cooldown_warning,
            Severity.ALARM: self.cfg.cooldown_alarm,
        }[sev]
        if t - self._last_fired[sev] < cooldown:
            return None
        self._last_fired[sev] = t

        text = self._pick_message(sev, state)
        self.history.append((t, sev, text))
        try:
            self._q.put_nowait((sev, text))
        except queue.Full:
            pass                         # audio is behind; drop, never block
        return sev

    # ------------------------------------------------------------------
    def _severity(self, t: float, state) -> Severity:
        if state.level is AlertLevel.AWAKE:
            self._drowsy_since = None
            return Severity.NONE

        if state.level is AlertLevel.WARNING:
            self._drowsy_since = None
            return Severity.WARNING

        # DROWSY
        if self._drowsy_since is None:
            self._drowsy_since = t
        sustained = t - self._drowsy_since >= self.cfg.escalate_after_s
        return Severity.ALARM if sustained or state.microsleeps >= 3 \
            else Severity.ALARM

    def _pick_message(self, sev: Severity, state) -> str:
        pool = MESSAGES.get(sev)
        if not pool:
            return ""
        # alternate so the same sentence is never heard twice in a row
        text = pool[self._msg_idx % len(pool)]
        if text == self._last_text and len(pool) > 1:
            self._msg_idx += 1
            text = pool[self._msg_idx % len(pool)]
        self._msg_idx += 1
        self._last_text = text
        return text

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sev, text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                path = self._chimes.get(sev)
                if path:
                    _play_wav(path)
                if text:
                    if self._tts is not None:
                        self._tts.say(text)
                        self._tts.runAndWait()
                    else:
                        print(f"[ALERT {sev.name}] {text}")
            except Exception:
                pass                     # a failing speaker must not kill the run
            finally:
                self._q.task_done()

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2.0)
        try:
            for p in self._chimes.values():
                os.remove(p)
            os.rmdir(self._tmp)
        except Exception:
            pass

    def report(self) -> str:
        if not self.history:
            return "No alert was raised."
        lines = [f"{len(self.history)} alert(s):"]
        for t, sev, text in self.history:
            lines.append(f"  t={t:7.2f}s  {sev.name:<8} {text}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from drowsiness_state import DrowsinessMonitor, Config, EyeState

    print("Simulating: 12s awake, then repeated 0.8s closures.\n"
          "Listen for a soft chime then a spoken message.\n")

    alerts = AlertManager()
    monitor = DrowsinessMonitor(Config())
    t = 0.0

    def feed(state, seconds):
        global t
        for _ in range(int(seconds * 25)):
            st = monitor.update(t, state, False)
            fired = alerts.notify(t, st)
            if fired:
                print(f"t={t:6.2f}s  {st.level.name:<8} -> {fired.name}")
            t += 1 / 25

    feed(EyeState.OPEN, 12)
    for _ in range(4):
        feed(EyeState.CLOSED, 0.8)
        feed(EyeState.OPEN, 3)

    time.sleep(4)          # let the worker finish speaking
    print("\n" + alerts.report())
    alerts.close()