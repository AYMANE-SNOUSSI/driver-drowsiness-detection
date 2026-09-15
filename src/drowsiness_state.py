"""
Temporal drowsiness state machine.

Detector-agnostic: takes per-frame observations (eye state, mouth state) and
produces a drowsiness decision based on time, not on single frames.

Works with YOLO, MediaPipe, or any other frontend. That is the point: the
same decision layer is used for both, so the comparison is fair.

Metrics implemented
-------------------
PERCLOS   : fraction of time eyes are closed over a sliding window.
            NHTSA reference measure, defined over 60s. >0.15 = drowsiness.
Microsleep: continuous eye closure >500ms. Needs no window at all.
Yawn      : mouth open continuously for >3s. A yawn is a *duration*, not an
            image — this is why the yawn class was removed from the detector
            and rebuilt here.
Blink rate: blinks per minute. Drops sharply before sleep onset.

All timing is in seconds from video/wall timestamps, NOT frame counts, so
results are independent of the detector's frame rate.

PERCLOS validity
----------------
PERCLOS is only meaningful once enough time has been observed. On a 24s clip
the 60s window never fills, so the value is not comparable to the NHTSA
reference. The monitor tracks how much of the window is covered and refuses
to apply the PERCLOS threshold below `perclos_min_window_s`. Microsleep
detection still works throughout, because a 0.9s closure needs no window.

Leaving the alarm
-----------------
An earlier version latched: once DROWSY, with PERCLOS unavailable, there was
no path back. That is a lock with no key — on a short clip the alarm stayed
on even after the driver clearly recovered.

Real driver-monitoring systems do latch, but they release on an explicit
event: the driver acknowledges, or the vehicle stops. Two paths are provided
here:

  * timeout  — no new microsleep or long closure for `alarm_release_s`
               downgrades DROWSY to WARNING, and twice that returns to AWAKE.
  * manual   — acknowledge() clears the alarm and the event history, as a
               dashboard button would.

The alarm never downgrades while events keep occurring.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EyeState(Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"      # no eye detected this frame


class AlertLevel(Enum):
    AWAKE = 0
    WARNING = 1
    DROWSY = 2


@dataclass
class Config:
    # --- PERCLOS ---
    window_s: float = 60.0               # NHTSA reference window
    perclos_min_window_s: float = 30.0   # below this, PERCLOS is not trusted
    perclos_warn: float = 0.12
    perclos_alarm: float = 0.15
    perclos_release: float = 0.08        # hysteresis: exit below this

    # --- microsleep ---
    microsleep_s: float = 0.5            # closure longer than this = microsleep
    microsleep_alarm_s: float = 1.5      # a single closure this long = alarm
    microsleep_count_alarm: int = 3      # this many microsleeps in the window

    # --- leaving the alarm ---
    alarm_release_s: float = 45.0        # quiet time before downgrading

    # --- blink ---
    blink_max_s: float = 0.4             # closure shorter than this = blink

    # --- yawn ---
    yawn_min_s: float = 3.0
    yawn_window_s: float = 300.0
    yawn_alarm_count: int = 3

    # --- robustness ---
    max_gap_s: float = 1.0               # hold last state this long when
                                         # detection is lost, then UNKNOWN
    min_coverage: float = 0.5            # need this share of valid samples


@dataclass
class State:
    level: AlertLevel = AlertLevel.AWAKE
    perclos: float = 0.0
    perclos_valid: bool = False          # is PERCLOS comparable to NHTSA?
    observed_s: float = 0.0              # span of data actually in the window
    coverage: float = 0.0                # share of that span with detections
    longest_closure_s: float = 0.0
    current_closure_s: float = 0.0
    microsleeps: int = 0
    blinks_per_min: float = 0.0
    yawns_in_window: int = 0
    quiet_s: float = 0.0                 # time since the last fatigue event
    acknowledged: bool = False
    reasons: list = field(default_factory=list)


class DrowsinessMonitor:
    """
        m = DrowsinessMonitor()
        st = m.update(t, EyeState.CLOSED, mouth_open=False)
        if st.level is AlertLevel.DROWSY:
            trigger_alarm()
        ...
        m.acknowledge(t)      # driver pressed the button
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self._samples = deque()          # (t, EyeState)
        self._blinks = deque()
        self._micro = deque()
        self._yawns = deque()

        self._closure_start: Optional[float] = None
        self._mouth_open_start: Optional[float] = None
        self._last_valid_t: Optional[float] = None
        self._last_valid_eye = EyeState.UNKNOWN
        self._last_event_t: Optional[float] = None   # last microsleep / long closure
        self._level = AlertLevel.AWAKE
        self._acked = False

    # ------------------------------------------------------------------
    def acknowledge(self, t: float) -> None:
        """Driver acknowledged the alarm: clear it and the event history.

        Clearing the history matters. Without it, the microsleeps still in
        the window would re-trigger the alarm on the very next frame, and the
        button would do nothing.
        """
        self._level = AlertLevel.AWAKE
        self._micro.clear()
        self._last_event_t = t
        self._acked = True

    # ------------------------------------------------------------------
    def update(self, t: float, eye: EyeState, mouth_open: bool) -> State:
        """t: timestamp in seconds (video PTS, or time.perf_counter())."""
        cfg = self.cfg

        eye = self._fill_gap(t, eye)
        self._samples.append((t, eye))
        self._trim(self._samples, t - cfg.window_s)
        self._trim(self._blinks, t - 60.0)
        self._trim(self._micro, t - cfg.window_s)
        self._trim(self._yawns, t - cfg.yawn_window_s)

        self._track_closure(t, eye)
        self._track_mouth(t, mouth_open)

        perclos, coverage, observed = self._perclos()
        current = (t - self._closure_start) if self._closure_start else 0.0
        if current >= cfg.microsleep_s:
            self._last_event_t = t       # an ongoing long closure is an event

        quiet = (t - self._last_event_t) if self._last_event_t is not None else 0.0

        st = State(
            perclos=perclos,
            perclos_valid=(observed >= cfg.perclos_min_window_s
                           and coverage >= cfg.min_coverage),
            observed_s=observed,
            coverage=coverage,
            current_closure_s=current,
            longest_closure_s=self._longest_closure(),
            microsleeps=len(self._micro),
            blinks_per_min=float(len(self._blinks)),
            yawns_in_window=len(self._yawns),
            quiet_s=quiet,
            acknowledged=self._acked,
        )
        st.level = self._decide(t, st)
        st.reasons = self._reasons(st)
        return st

    # ------------------------------------------------------------------
    def _fill_gap(self, t: float, eye: EyeState) -> EyeState:
        """Hold the last known state briefly when detection drops out.

        A single missed frame must not reset the closure timer — that was the
        flaw in the naive start_time approach.
        """
        if eye is not EyeState.UNKNOWN:
            self._last_valid_t = t
            self._last_valid_eye = eye
            return eye
        if self._last_valid_t is not None and t - self._last_valid_t <= self.cfg.max_gap_s:
            return self._last_valid_eye
        return EyeState.UNKNOWN

    def _track_closure(self, t: float, eye: EyeState) -> None:
        if eye is EyeState.CLOSED:
            if self._closure_start is None:
                self._closure_start = t
            return
        if self._closure_start is not None:
            dur = t - self._closure_start
            if dur <= self.cfg.blink_max_s:
                self._blinks.append(t)
            elif dur >= self.cfg.microsleep_s:
                self._micro.append(t)
                self._last_event_t = t
                self._acked = False      # a new event invalidates the ack
            self._closure_start = None

    def _track_mouth(self, t: float, mouth_open: bool) -> None:
        if mouth_open:
            if self._mouth_open_start is None:
                self._mouth_open_start = t
            return
        if self._mouth_open_start is not None:
            if t - self._mouth_open_start >= self.cfg.yawn_min_s:
                self._yawns.append(t)
            self._mouth_open_start = None

    # ------------------------------------------------------------------
    def _perclos(self):
        """Returns (perclos, coverage, observed_span_seconds).

        Time-weighted, not sample-counted: a detector at 10 FPS and one at
        30 FPS give the same value on the same footage.
        """
        if len(self._samples) < 2:
            return 0.0, 0.0, 0.0
        closed = valid = total = 0.0
        prev_t, prev_s = self._samples[0]
        for t, s in list(self._samples)[1:]:
            dt = t - prev_t
            total += dt
            if prev_s is not EyeState.UNKNOWN:
                valid += dt
                if prev_s is EyeState.CLOSED:
                    closed += dt
            prev_t, prev_s = t, s
        if valid <= 0:
            return 0.0, 0.0, total
        return closed / valid, valid / total, total

    def _longest_closure(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        longest = run = 0.0
        prev_t, prev_s = self._samples[0]
        for t, s in list(self._samples)[1:]:
            if prev_s is EyeState.CLOSED:
                run += t - prev_t
                longest = max(longest, run)
            else:
                run = 0.0
            prev_t, prev_s = t, s
        return longest

    @staticmethod
    def _trim(dq: deque, cutoff: float) -> None:
        while dq:
            item = dq[0]
            ts = item[0] if isinstance(item, tuple) else item
            if ts < cutoff:
                dq.popleft()
            else:
                break

    # ------------------------------------------------------------------
    def _decide(self, t: float, st: State) -> AlertLevel:
        """Hysteresis: harder to leave an alert state than to enter it.

        Without it the system oscillates around the threshold and the alarm
        flickers — useless, and dangerous, in a real vehicle.
        """
        cfg = self.cfg

        # --- signals that need no window ---
        if st.current_closure_s >= cfg.microsleep_alarm_s:
            self._level = AlertLevel.DROWSY
            return self._level
        if st.microsleeps >= cfg.microsleep_count_alarm:
            self._level = AlertLevel.DROWSY
            return self._level

        # --- window-based signals, only when the window is trustworthy ---
        if st.perclos_valid:
            if self._level is AlertLevel.DROWSY:
                if st.perclos < cfg.perclos_release:
                    self._level = AlertLevel.AWAKE
                return self._level
            if st.perclos >= cfg.perclos_alarm or st.yawns_in_window >= cfg.yawn_alarm_count:
                self._level = AlertLevel.DROWSY
            elif st.perclos >= cfg.perclos_warn:
                self._level = AlertLevel.WARNING
            else:
                self._level = AlertLevel.AWAKE
            return self._level

        # --- fallback while the window fills ---
        if self._level is AlertLevel.DROWSY:
            # the only way out without PERCLOS: sustained quiet
            if st.quiet_s >= 2 * cfg.alarm_release_s:
                self._level = AlertLevel.AWAKE
            elif st.quiet_s >= cfg.alarm_release_s:
                self._level = AlertLevel.WARNING
            return self._level

        if st.longest_closure_s >= cfg.microsleep_s or st.microsleeps:
            self._level = AlertLevel.WARNING
        else:
            self._level = AlertLevel.AWAKE
        return self._level

    def _reasons(self, st: State) -> list:
        cfg, out = self.cfg, []
        if st.current_closure_s >= cfg.microsleep_alarm_s:
            out.append(f"eyes closed {st.current_closure_s:.1f}s")
        if st.microsleeps:
            out.append(f"{st.microsleeps} microsleep(s)")
        if st.perclos_valid:
            if st.perclos >= cfg.perclos_warn:
                out.append(f"PERCLOS {st.perclos:.0%}")
        elif st.observed_s > 0:
            out.append(f"PERCLOS n/a ({st.observed_s:.0f}s / "
                       f"{cfg.perclos_min_window_s:.0f}s)")
        if st.level is not AlertLevel.AWAKE and st.quiet_s > 5:
            out.append(f"quiet {st.quiet_s:.0f}s")
        if st.acknowledged:
            out.append("acknowledged")
        if st.coverage < cfg.min_coverage and st.observed_s > 1:
            out.append(f"low coverage {st.coverage:.0%}")
        if st.yawns_in_window:
            out.append(f"{st.yawns_in_window} yawn(s)")
        return out


# ----------------------------------------------------------------------
def aggregate_eye_boxes(labels) -> EyeState:
    """Turn a frame's detections into a single eye state.

    The detector may return 0, 1 or 2 eyes. Rule: if any eye is closed and
    none is open, the driver's eyes are closed. Conservative on purpose —
    missing a closed eye is worse than a false warning.
    """
    labels = list(labels)
    closed = sum(1 for l in labels if l == "eye_closed")
    opened = sum(1 for l in labels if l == "eye_open")
    if closed and not opened:
        return EyeState.CLOSED
    if opened:
        return EyeState.OPEN
    return EyeState.UNKNOWN


# ----------------------------------------------------------------------
if __name__ == "__main__":
    def scenario(label, script, ack_at=None, cfg=None):
        m = DrowsinessMonitor(cfg or Config())
        t, st, marks = 0.0, None, []
        prev = None
        for state, seconds in script:
            for _ in range(int(seconds * 25)):
                if ack_at is not None and abs(t - ack_at) < 0.02:
                    m.acknowledge(t)
                    marks.append(f"ack@{t:.0f}s")
                st = m.update(t, state, False)
                if st.level is not prev:
                    marks.append(f"{st.level.name}@{t:.0f}s")
                    prev = st.level
                t += 1 / 25
        print(f"{label:<40} {' -> '.join(marks)}")

    print("leaving the alarm\n" + "-" * 78)
    scenario("3 microsleeps then 2 min calm",
             [(EyeState.OPEN, 5)]
             + [(EyeState.CLOSED, .8), (EyeState.OPEN, 2)] * 3
             + [(EyeState.OPEN, 120)])
    scenario("3 microsleeps, calm, driver acknowledges",
             [(EyeState.OPEN, 5)]
             + [(EyeState.CLOSED, .8), (EyeState.OPEN, 2)] * 3
             + [(EyeState.OPEN, 30)],
             ack_at=20.0)
    scenario("microsleeps that keep coming",
             [(EyeState.OPEN, 5)]
             + [(EyeState.CLOSED, .8), (EyeState.OPEN, 8)] * 8)