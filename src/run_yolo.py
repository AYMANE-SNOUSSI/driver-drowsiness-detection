"""
YOLO frontend + temporal decision layer, run over a video file.

Outputs
-------
runs/<tag>_annotated.mp4   video with live PERCLOS / alert overlay
runs/<tag>_metrics.csv     one row per frame, for later analysis
stdout                     latency summary (p50 / p95 / max)

Why a video file and not the webcam
-----------------------------------
The webcam gives a different recording every run, so YOLO and MediaPipe
could never be compared on equal footing. A fixed video makes the two
frontends see exactly the same frames.
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

from drowsiness_state import (
    DrowsinessMonitor, Config, EyeState, AlertLevel, aggregate_eye_boxes,
)

COLORS = {
    AlertLevel.AWAKE:   (0, 200, 0),
    AlertLevel.WARNING: (0, 165, 255),
    AlertLevel.DROWSY:  (0, 0, 255),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/drowsy_v4.pt")
    p.add_argument("--video", default="data/essay3.mp4")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--tag", default="yolo")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--device", default="cpu", help="cpu or 0 for GPU")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def draw(frame, st, fps, latency_ms):
    h, w = frame.shape[:2]
    color = COLORS[st.level]

    cv2.rectangle(frame, (0, 0), (w, 92), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)

    cv2.putText(frame, st.level.name, (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"PERCLOS {st.perclos:6.1%}   closure {st.current_closure_s:4.1f}s"
                       f"   yawns {st.yawns_in_window}",
                (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.putText(frame, f"{fps:5.1f} FPS   {latency_ms:5.1f} ms   coverage {st.coverage:.0%}",
                (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # PERCLOS bar with the alarm threshold marked
    bar_w, x, y = 260, w - 280, 20
    cv2.rectangle(frame, (x, y), (x + bar_w, y + 14), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(bar_w * min(st.perclos, 1.0)), y + 14), color, -1)
    thr = x + int(bar_w * 0.15)
    cv2.line(frame, (thr, y - 3), (thr, y + 17), (255, 255, 255), 1)

    if st.reasons:
        cv2.putText(frame, " | ".join(st.reasons[:2]), (14, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if not os.path.exists(a.model):
        raise SystemExit(f"Model not found: {a.model}")
    if not os.path.exists(a.video):
        raise SystemExit(f"Video not found: {a.video}")

    model = YOLO(a.model)
    names = model.names
    print(f"Model classes: {names}")

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit("Cannot open video")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h}  {src_fps:.1f} fps  {n_frames} frames  "
          f"({n_frames / src_fps:.1f}s)")

    out_path = os.path.join(a.out_dir, f"{a.tag}_annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps, (w, h))
    csv_path = os.path.join(a.out_dir, f"{a.tag}_metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_file)
    csv_w.writerow(["frame", "t_video", "eye_state", "mouth_open", "level",
                    "perclos", "closure_s", "yawns", "coverage", "latency_ms"])

    monitor = DrowsinessMonitor(Config())
    latencies, idx, disp_fps = [], 0, 0.0
    wall_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # VIDEO timestamp, not wall clock. Processing may run slower or
        # faster than real time; PERCLOS must reflect the driver's time,
        # not the machine's.
        t_video = idx / src_fps

        t0 = time.perf_counter()
        res = model.predict(frame, conf=a.conf, device=a.device,
                            verbose=False)[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        labels = [names[int(b.cls[0])] for b in res.boxes]
        eye = aggregate_eye_boxes(labels)
        mouth_open = "mouth_open" in labels

        st = monitor.update(t_video, eye, mouth_open)

        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            lab = names[int(b.cls[0])]
            c = (0, 0, 255) if lab == "eye_closed" else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"{lab} {float(b.conf[0]):.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

        if idx % 10 == 0:
            disp_fps = 1000 / max(np.mean(latencies[-30:]), 1e-6)

        frame = draw(frame, st, disp_fps, latency_ms)
        writer.write(frame)

        csv_w.writerow([idx, f"{t_video:.3f}", eye.value, int(mouth_open),
                        st.level.name, f"{st.perclos:.4f}",
                        f"{st.current_closure_s:.3f}", st.yawns_in_window,
                        f"{st.coverage:.3f}", f"{latency_ms:.2f}"])

        if not a.no_display:
            cv2.imshow("drowsiness (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        idx += 1
        if n_frames and idx % 100 == 0:
            print(f"  {idx}/{n_frames}  ({100 * idx / n_frames:.0f}%)")

    cap.release()
    writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    a_ = np.array(latencies)
    wall = time.perf_counter() - wall_start
    print(f"\n=== {a.tag.upper()} on {a.device} ===")
    print(f"frames            : {idx}")
    print(f"latency p50       : {np.percentile(a_, 50):.2f} ms")
    print(f"latency p95       : {np.percentile(a_, 95):.2f} ms")
    print(f"latency max       : {a_.max():.2f} ms")
    print(f"throughput        : {idx / wall:.1f} FPS")
    print(f"real-time capable : {'yes' if np.percentile(a_, 95) < 1000 / src_fps else 'NO'}"
          f"  (budget {1000 / src_fps:.1f} ms at {src_fps:.0f} fps)")
    print(f"\nvideo -> {out_path}")
    print(f"csv   -> {csv_path}")


if __name__ == "__main__":
    main()