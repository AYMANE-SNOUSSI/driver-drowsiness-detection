"""
Measure ONE backend, alone in its own process.

Why one at a time
-----------------
Loading PyTorch, ONNX Runtime and OpenVINO in the same process puts three
thread pools on the same 4 cores. They contend, and the result is bimodal
latency: OpenVINO showed a 56 ms median with a 563 ms p95, which is thread
contention, not the backend.

Running each backend in a separate process removes the contention. The cost
is that thermal state can differ between runs, so the script prints a drift
check: the first and last 15 frames are compared, and the run is flagged if
the machine changed speed.

Protocol
--------
Run each backend twice, alternating, and keep the median of the medians:

    python src/bench_one.py --backend pytorch
    python src/bench_one.py --backend onnx
    python src/bench_one.py --backend openvino
    python src/bench_one.py --backend openvino
    python src/bench_one.py --backend onnx
    python src/bench_one.py --backend pytorch

Close other applications, plug in power, set Windows to High Performance.
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True,
                   choices=["pytorch", "onnx", "openvino"])
    p.add_argument("--model", default="models/drowsy_v4.pt")
    p.add_argument("--video", default="data/essay3.mp4")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--max-width", type=int, default=960)
    p.add_argument("--fps-target", type=float, default=25.0)
    p.add_argument("--skip-first", type=int, default=0,
                   help="drop the first N timed frames (startup ramp)")
    return p.parse_args()


def resolve(backend, model_pt):
    if backend == "pytorch":
        return model_pt, {}
    if backend == "onnx":
        path = model_pt.replace(".pt", ".onnx")
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing. Run bench_backends.py once to export.")
        return path, {"task": "detect"}
    path = model_pt.replace(".pt", "_openvino_model")
    if not os.path.isdir(path):
        raise SystemExit(f"{path} missing. Run bench_backends.py once to export.")
    return path, {"task": "detect"}


def main():
    a = parse_args()
    path, kw = resolve(a.backend, a.model)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {a.video}")
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = min(1.0, a.max_width / sw)
    frames = []
    while len(frames) < a.frames + a.warmup:
        ok, f = cap.read()
        if not ok:
            break
        if scale < 1.0:
            f = cv2.resize(f, (int(f.shape[1] * scale), int(f.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if len(frames) < a.warmup + 30:
        raise SystemExit("Not enough frames")

    warm, timed = frames[:a.warmup], frames[a.warmup:]
    timed_all = timed
    model = YOLO(path, **kw)

    for f in warm:
        model.predict(f, imgsz=a.imgsz, conf=a.conf, device="cpu", verbose=False)

    lat = []
    for f in timed_all[a.skip_first:]:
        t0 = time.perf_counter()
        model.predict(f, imgsz=a.imgsz, conf=a.conf, device="cpu", verbose=False)
        lat.append((time.perf_counter() - t0) * 1000)

    v = np.array(lat)
    p25, p50, p75 = np.percentile(v, [25, 50, 75])
    first = float(np.median(v[:15]))
    last = float(np.median(v[-15:]))
    drift = last / first - 1.0
    budget = 1000.0 / a.fps_target

    print(f"\n=== {a.backend.upper()}  ({len(timed)} frames, "
          f"{timed[0].shape[1]}x{timed[0].shape[0]}) ===")
    print(f"p50        {p50:8.1f} ms")
    print(f"p95        {np.percentile(v, 95):8.1f} ms")
    print(f"max        {v.max():8.1f} ms")
    print(f"IQR        {p75 - p25:8.1f} ms   (spread; large = unstable machine)")
    print(f"FPS (p50)  {1000 / p50:8.1f}")
    print(f"budget     {budget:8.1f} ms at {a.fps_target:.0f} fps  ->  "
          f"{'OK' if np.percentile(v, 95) < budget else 'too slow'}")
    print(f"drift      {first:.1f} -> {last:.1f} ms ({drift:+.0%})   "
          f"{'unstable, rerun' if abs(drift) > 0.15 else 'stable'}")


if __name__ == "__main__":
    main()