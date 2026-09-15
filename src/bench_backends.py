"""
Benchmark YOLO backends on CPU: PyTorch vs ONNX Runtime vs OpenVINO.

Why this version exists
-----------------------
The first version ran each backend to completion, one after the other. On a
laptop CPU that is a broken protocol: the chip heats up and downclocks, so
whichever backend runs last is penalised. It produced a p50 of 62 ms with a
p95 of 861 ms for the same model doing the same work every frame — a range
that no backend can explain, only thermal drift can.

This version interleaves. Every frame is run through every backend, and the
order rotates frame to frame, so throttling hits all of them equally.

Two other fixes:
  * equivalence is checked with IoU matching, not exact pixel coordinates.
    Sub-pixel shifts after export are expected and must not count as errors.
  * a drift check re-measures the baseline at the end. If the machine slowed
    down during the run, the results are flagged as unreliable.

Before running
--------------
Close browsers and editors, plug the laptop in, and set the Windows power
plan to high performance. A benchmark on a throttled, loaded machine
measures the machine, not the model.

Install
-------
    pip install onnx onnxruntime
    pip install openvino
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/drowsy_v4.pt")
    p.add_argument("--video", default="data/essay3.mp4")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--max-width", type=int, default=960)
    p.add_argument("--fps-target", type=float, default=25.0)
    p.add_argument("--iou", type=float, default=0.90,
                   help="IoU above which two boxes count as the same")
    p.add_argument("--skip-openvino", action="store_true")
    return p.parse_args()


def load_frames(path, n, max_width):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {path}")
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = min(1.0, max_width / sw)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if scale < 1.0:
            f = cv2.resize(f, (int(f.shape[1] * scale), int(f.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit("No frames decoded")
    print(f"Loaded {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    return frames


def detections(res, names):
    return [(names[int(b.cls[0])], [float(v) for v in b.xyxy[0]],
             float(b.conf[0])) for b in res.boxes]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compare(ref_frames, other_frames, iou_thr):
    """Matched / missed / extra detections against the baseline."""
    matched = missed = extra = 0
    conf_delta = []
    for ref, oth in zip(ref_frames, other_frames):
        used = set()
        for cls_r, box_r, conf_r in ref:
            best, best_i = None, 0.0
            for j, (cls_o, box_o, conf_o) in enumerate(oth):
                if j in used or cls_o != cls_r:
                    continue
                v = iou(box_r, box_o)
                if v > best_i:
                    best, best_i = j, v
            if best is not None and best_i >= iou_thr:
                used.add(best)
                matched += 1
                conf_delta.append(abs(oth[best][2] - conf_r))
            else:
                missed += 1
        extra += len(oth) - len(used)
    total = matched + missed
    return {
        "recall": matched / total if total else 1.0,
        "missed": missed,
        "extra": extra,
        "conf_mad": float(np.mean(conf_delta)) if conf_delta else 0.0,
    }


def main():
    a = parse_args()
    if not os.path.exists(a.model):
        raise SystemExit(f"Model not found: {a.model}")

    frames = load_frames(a.video, a.frames + a.warmup, a.max_width)
    warm, timed = frames[:a.warmup], frames[a.warmup:]
    budget = 1000.0 / a.fps_target

    # ---------- build the backends ----------
    backends = {}
    base = YOLO(a.model)
    backends["PyTorch"] = base
    print(f"Classes: {base.names}")

    onnx_path = a.model.replace(".pt", ".onnx")
    if not os.path.exists(onnx_path):
        print("exporting ONNX ...")
        base.export(format="onnx", imgsz=a.imgsz, opset=12,
                    simplify=True, dynamic=False)
    if os.path.exists(onnx_path):
        backends["ONNX"] = YOLO(onnx_path, task="detect")

    if not a.skip_openvino:
        ov_dir = a.model.replace(".pt", "_openvino_model")
        if not os.path.isdir(ov_dir):
            try:
                print("exporting OpenVINO ...")
                base.export(format="openvino", imgsz=a.imgsz)
            except Exception as e:
                print(f"OpenVINO export skipped: {type(e).__name__}: {e}")
        if os.path.isdir(ov_dir):
            try:
                backends["OpenVINO"] = YOLO(ov_dir, task="detect")
            except Exception as e:
                print(f"OpenVINO load skipped: {type(e).__name__}")

    order = list(backends)
    print(f"\nBackends: {', '.join(order)}")

    # ---------- warm up everything ----------
    print("warming up ...")
    for name in order:
        for f in warm:
            backends[name].predict(f, imgsz=a.imgsz, conf=a.conf,
                                   device="cpu", verbose=False)

    # ---------- baseline drift probe, before ----------
    def probe():
        t = []
        for f in timed[:10]:
            t0 = time.perf_counter()
            backends["PyTorch"].predict(f, imgsz=a.imgsz, conf=a.conf,
                                        device="cpu", verbose=False)
            t.append((time.perf_counter() - t0) * 1000)
        return float(np.median(t))

    probe_before = probe()

    # ---------- interleaved timing ----------
    print(f"timing {len(timed)} frames, interleaved ...")
    lat = {k: [] for k in order}
    dets = {k: [] for k in order}

    for i, f in enumerate(timed):
        rotated = order[i % len(order):] + order[:i % len(order)]
        for name in rotated:
            m = backends[name]
            t0 = time.perf_counter()
            res = m.predict(f, imgsz=a.imgsz, conf=a.conf,
                            device="cpu", verbose=False)[0]
            lat[name].append((time.perf_counter() - t0) * 1000)
            dets[name].append(detections(res, m.names))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(timed)}")

    probe_after = probe()
    drift = probe_after / probe_before - 1.0

    # ---------- report ----------
    print(f"\n{'backend':<12}{'p50':>9}{'p95':>9}{'max':>9}{'IQR':>9}"
          f"{'speedup':>10}{f'{a.fps_target:.0f}fps?':>8}")
    print("-" * 66)
    ref = float(np.percentile(lat["PyTorch"], 50))
    for name in order:
        v = np.array(lat[name])
        p25, p50, p75 = np.percentile(v, [25, 50, 75])
        print(f"{name:<12}{p50:9.1f}{np.percentile(v,95):9.1f}{v.max():9.1f}"
              f"{p75-p25:9.1f}{ref/p50:9.2f}x"
              f"{('yes' if np.percentile(v,95) < budget else 'no'):>8}")
    print(f"\nall times in ms, budget {budget:.0f} ms at {a.fps_target:.0f} fps")

    print(f"\nThermal drift check: baseline {probe_before:.1f} ms before, "
          f"{probe_after:.1f} ms after ({drift:+.0%})")
    if abs(drift) > 0.15:
        print("  WARNING: the machine changed speed during the run.")
        print("  Close other applications, plug in power, and run again.")
        print("  Do not quote these numbers.")
    else:
        print("  OK: the machine stayed stable, the comparison is fair.")

    print(f"\nEquivalence with PyTorch (IoU >= {a.iou}):")
    n_ref = sum(len(d) for d in dets["PyTorch"])
    print(f"  baseline detections: {n_ref}")
    for name in order:
        if name == "PyTorch":
            continue
        r = compare(dets["PyTorch"], dets[name], a.iou)
        print(f"  {name:<10} matched {r['recall']:6.1%}   "
              f"missed {r['missed']:<4} extra {r['extra']:<4} "
              f"mean |Δconf| {r['conf_mad']:.4f}")
    print("\nA few missed or extra boxes near the confidence threshold are\n"
          "expected. A matched rate below ~95% means the export changed the\n"
          "model, and the speedup is not free.")


if __name__ == "__main__":
    main()
    