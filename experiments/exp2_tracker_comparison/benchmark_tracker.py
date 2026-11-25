"""
Thực nghiệm 2: Đánh giá & So sánh Giải thuật Tracking Đa Đối tượng (MOT)
Đánh giá song song trên cả 2 mô hình triển khai của hệ thống:
1. YOLO26s (ưu tiên OpenVINO FP16 trên Intel Iris Xe iGPU; dự phòng PyTorch CPU)
2. YOLO26m (ưu tiên OpenVINO FP16 trên Intel Iris Xe iGPU; dự phòng PyTorch CPU)

Lý do chạy trên OpenVINO iGPU (khi khả dụng):
- Phản ánh đúng pipeline vận hành thực tế của hệ thống (Detector + MOT Tracker).
- Độ trễ suy luận trên iGPU cực ổn định (biến thiên < 1.5 ms), loại trừ nhiễu thermal throttling của CPU
  để đo đạc chính xác chi phí bổ sung (Tracking Overhead) của BoT-SORT vs ByteTrack.

Dữ liệu thử nghiệm: dataset_test/test_exp.mp4 (nạp trước vào RAM để loại trừ I/O đĩa)
So sánh 3 phương pháp cho mỗi mô hình:
1. Pure YOLO (Không Tracking — Baseline phát hiện từng khung hình)
2. BoT-SORT (Kalman Filter + Camera Motion Compensation GMC sparseOptFlow)
3. ByteTrack (Liên kết 2 tầng điểm tin cậy theo IoU + Kalman Filter — Đề xuất)
"""

import os
import sys
import time
import tempfile
import argparse
import numpy as np
import cv2
import pandas as pd
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EXP2_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(EXP2_DIR, "..", ".."))
RESULTS_DIR = os.path.join(EXP2_DIR, "results")

if not os.environ.get("YOLO_CONFIG_DIR"):
    _tmp_cfg = os.path.join(tempfile.gettempdir(), "railsafe_ultralytics_cfg")
    try:
        os.makedirs(_tmp_cfg, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = _tmp_cfg
    except Exception:
        _local_cfg = os.path.join(EXP2_DIR, ".ultralytics_cfg")
        os.makedirs(_local_cfg, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = _local_cfg

from ultralytics import YOLO

# Patch AutoBackend để Ultralytics nhận diện chuẩn thư mục model_openvino
try:
    from ultralytics.nn import autobackend
    _orig_model_type = autobackend.AutoBackend._model_type

    def _patched_model_type(p, dnn=False):
        fmt = _orig_model_type(p, dnn)
        if fmt is None and "openvino" in str(p).lower():
            return "openvino"
        return fmt

    autobackend.AutoBackend._model_type = staticmethod(_patched_model_type)
except Exception:
    pass


def first_existing(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0]


def detect_best_backend():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda", "cuda"
    except Exception:
        pass
    try:
        import openvino as ov
        if "GPU" in ov.Core().available_devices:
            return "openvino", "intel:gpu"
    except Exception:
        pass
    return "pytorch", "cpu"


def resolve_models(backend="auto"):
    if backend == "auto":
        backend, device = detect_best_backend()
    elif backend == "openvino":
        device = "intel:gpu"
    elif backend == "cuda":
        device = "cuda"
    else:
        device = "cpu"

    if backend == "openvino":
        models = {
            "YOLO26s": first_existing([
                os.path.join(BASE_DIR, "models", "yolo26s", "model_openvino"),
                os.path.join(BASE_DIR, "models", "yolo26s", "model_pytorch", "train-railway-yolo26s-all.pt"),
            ]),
            "YOLO26m": first_existing([
                os.path.join(BASE_DIR, "models", "yolo26m", "model_openvino"),
                os.path.join(BASE_DIR, "models", "yolo26m", "model_pytorch", "yolo26m.pt"),
            ]),
        }
    else:
        models = {
            "YOLO26s": first_existing([
                os.path.join(BASE_DIR, "models", "yolo26s", "model_pytorch", "train-railway-yolo26s-all.pt"),
                os.path.join(BASE_DIR, "models", "yolo26s", "model_pytorch", "yolo26s.pt"),
                os.path.join(BASE_DIR, "research", "finetune", "train_yolo26s", "yolo26s.pt"),
            ]),
            "YOLO26m": first_existing([
                os.path.join(BASE_DIR, "models", "yolo26m", "model_pytorch", "yolo26m.pt"),
                os.path.join(BASE_DIR, "research", "finetune", "train_yolo26m", "yolo26m.pt"),
                os.path.join(BASE_DIR, "Fine-tune-Dataset-Lam", "Fine-tune-Dataset-Lam", "model", "best.pt"),
            ]),
        }
    return models, backend, device


def reset_model_trackers(model):
    """Đặt lại hoàn toàn trạng thái tracker sau bước warmup mà vẫn giữ nguyên mô hình đã nạp."""
    try:
        if hasattr(model, "predictor") and hasattr(model.predictor, "trackers"):
            for t in model.predictor.trackers:
                if hasattr(t, "reset"):
                    t.reset()
    except Exception:
        pass


def evaluate_tracking_mode(
    model, mode, frames, tracker_cfg=None, warmup_iters=10, conf=0.25, imgsz=640, device="intel:gpu"
):
    cfg = tracker_cfg if (tracker_cfg and os.path.exists(tracker_cfg)) else "bytetrack.yaml"

    # Reset predictor/trackers sạch trước mỗi chế độ trên cùng instance model đã ổn định xung nhịp
    if hasattr(model, "predictor") and model.predictor is not None:
        reset_model_trackers(model)
        if hasattr(model.predictor, "trackers"):
            delattr(model.predictor, "trackers")

    warmup_frame = frames[0]
    for _ in range(warmup_iters):
        if mode == "none":
            _ = model.predict(warmup_frame, imgsz=imgsz, device=device, verbose=False, conf=conf)
        elif mode == "botsort":
            _ = model.track(warmup_frame, tracker="botsort.yaml", imgsz=imgsz, device=device, verbose=False, conf=conf, persist=True)
        elif mode == "bytetrack":
            _ = model.track(warmup_frame, tracker=cfg, imgsz=imgsz, device=device, verbose=False, conf=conf, persist=True)

    reset_model_trackers(model)

    latencies = []
    unique_ids = set()
    id_frame_history = defaultdict(list)

    for f_idx, frame in enumerate(frames):
        t0 = time.perf_counter()

        if mode == "none":
            results = model.predict(frame, imgsz=imgsz, device=device, verbose=False, conf=conf)
        elif mode == "botsort":
            results = model.track(frame, tracker="botsort.yaml", imgsz=imgsz, device=device, verbose=False, conf=conf, persist=True)
        elif mode == "bytetrack":
            results = model.track(frame, tracker=cfg, imgsz=imgsz, device=device, verbose=False, conf=conf, persist=True)

        latencies.append((time.perf_counter() - t0) * 1000.0)

        if mode != "none" and results and results[0].boxes is not None and results[0].boxes.id is not None:
            current_ids = results[0].boxes.id.int().cpu().tolist()
            for tid in current_ids:
                unique_ids.add(tid)
                id_frame_history[tid].append(f_idx)

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    if mode == "none" or len(unique_ids) == 0:
        avg_lifespan_frames = 0.0
        fragmentations = 0
        dwell_status = "Không thể đo (Không có ID)" if mode == "none" else "Không phát hiện đối tượng"
    else:
        lifespans = [len(f_list) for f_list in id_frame_history.values()]
        avg_lifespan_frames = float(np.mean(lifespans))
        fragmentations = 0
        for f_list in id_frame_history.values():
            for i in range(1, len(f_list)):
                if f_list[i] - f_list[i - 1] > 1:
                    fragmentations += 1
        dwell_status = f"Khả thi (Lưu bình quân {avg_lifespan_frames:.1f} frames/ID)"

    return {
        "mode": mode,
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "fps": round(fps, 1),
        "total_unique_ids": len(unique_ids) if mode != "none" else "N/A",
        "avg_lifespan_frames": round(avg_lifespan_frames, 1),
        "id_fragmentations": fragmentations if mode != "none" else "N/A",
        "dwell_support": dwell_status,
    }


def generate_chart(df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    models = df["Model"].unique().tolist()
    methods = ["Pure YOLO (No Tracking)", "BoT-SORT", "ByteTrack (Đề xuất)"]
    colors = ["#64748b", "#f59e0b", "#0d828a"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(models))
    width = 0.25

    for idx, (m_label, col) in enumerate(zip(methods, colors)):
        vals, ov_vals = [], []
        for mod in models:
            sub = df[(df["Model"] == mod) & (df["Method"] == m_label)]
            vals.append(sub.iloc[0]["Latency_ms"] if not sub.empty else 0.0)
            ov_vals.append(sub.iloc[0]["Overhead_ms"] if not sub.empty else 0.0)
        offset = (idx - 1) * width
        b1 = axes[0].bar(x + offset, vals, width, label=m_label, color=col)
        for b in b1:
            h = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)

        if idx > 0:
            b2 = axes[1].bar(x + (idx - 1.5) * 0.32, ov_vals, 0.32, label=m_label, color=col)
            for b in b2:
                h = b.get_height()
                axes[1].text(b.get_x() + b.get_width() / 2, h + 0.15, f"{h:+.2f}ms", ha="center", va="bottom", fontsize=8.5)

    axes[0].set_title("Độ trễ trung bình theo Mô hình & Giải thuật Tracking (ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Latency (ms/frame)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("Chi phí tính toán bổ sung (Tracking Overhead so với Pure YOLO)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Overhead (ms/frame)")
    axes[1].axhline(0, color="#334155", linewidth=0.8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "chart_tracker_comparison.png"), dpi=160)
    plt.close(fig)


def run_experiment(video_path=None, num_frames=150, start_frame=0, backend="auto", imgsz=640, conf=0.25):
    models, resolved_backend, device = resolve_models(backend)

    print("=" * 104)
    print(" THỰC NGHIỆM 2: SO SÁNH GIẢI THUẬT TRACKING ĐA ĐỐI TƯỢNG (MOT) — YOLO26s & YOLO26m")
    print("=" * 104)

    if not video_path:
        for folder_name in ("dataset_test", "Dataset_test"):
            candidate = os.path.join(BASE_DIR, folder_name, "test_exp.mp4")
            if os.path.exists(candidate):
                video_path = candidate
                break
        else:
            video_path = os.path.join(BASE_DIR, "dataset_test", "test_exp.mp4")

    if not os.path.exists(video_path):
        print(f"[!] LỖI: Không tìm thấy video thực nghiệm tại: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"[!] LỖI: Không đọc được khung hình nào từ video {video_path}!")
        return None

    print(f"[*] Video thực nghiệm     : {video_path}")
    print(f"[*] Backend / Thiết bị    : {resolved_backend} ({device})")
    print(f"[*] Khung hình nạp RAM    : {len(frames)} frames (từ frame {start_frame}, imgsz={imgsz}, conf={conf})")

    tracker_cfg = os.path.join(EXP2_DIR, "bytetrack.yaml")
    if not os.path.exists(tracker_cfg):
        tracker_cfg = os.path.join(BASE_DIR, "configs", "bytetrack.yaml")

    all_rows = []
    mode_specs = [
        ("none", "Pure YOLO (No Tracking)"),
        ("botsort", "BoT-SORT"),
        ("bytetrack", "ByteTrack (Đề xuất)"),
    ]

    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"[!] Bỏ qua {model_name} (Không tìm thấy: {model_path})")
            continue

        print(f"\n>>> Đang đánh giá MOT với detector {model_name}: {model_path}")
        # Nạp mô hình 1 lần duy nhất và chạy warmup ổn định xung nhịp phần cứng trước khi đo cả 3 chế độ
        shared_model = YOLO(model_path)
        for _ in range(20):
            _ = shared_model.predict(frames[0], imgsz=imgsz, device=device, verbose=False, conf=conf)

        base_lat = None

        for mode_key, mode_label in mode_specs:
            print(f"    [*] Chạy {mode_label}...")
            res = evaluate_tracking_mode(
                shared_model, mode_key, frames,
                tracker_cfg=tracker_cfg, warmup_iters=10, conf=conf, imgsz=imgsz, device=device
            )
            if mode_key == "none":
                base_lat = res["avg_latency_ms"]
                overhead = 0.0
            else:
                overhead = round(res["avg_latency_ms"] - base_lat, 2)

            all_rows.append({
                "Model": model_name,
                "Method": mode_label,
                "Backend": f"{resolved_backend} ({device})",
                "Frames": len(frames),
                "Latency_ms": res["avg_latency_ms"],
                "P95_Latency_ms": res["p95_latency_ms"],
                "Overhead_ms": overhead,
                "FPS": res["fps"],
                "Unique_IDs": res["total_unique_ids"],
                "Avg_Lifespan_Frames": res["avg_lifespan_frames"],
                "ID_Fragmentations": res["id_fragmentations"],
                "Dwell_Support": res["dwell_support"],
            })

            print(
                f"        -> Avg={res['avg_latency_ms']}ms | P95={res['p95_latency_ms']}ms | "
                f"Overhead={overhead:+.2f}ms | FPS={res['fps']} | IDs={res['total_unique_ids']} | "
                f"Lifespan={res['avg_lifespan_frames']}f | Frag={res['id_fragmentations']}"
            )

    if not all_rows:
        return None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "tracker_comparison.csv")
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    generate_chart(df)

    print("\n" + "=" * 116)
    print(" BẢNG TỔNG KẾT THỰC NGHIỆM 2: SO SÁNH TRACKING MOT (YOLO26s & YOLO26m)")
    print("=" * 116)
    print(df.to_string(index=False))
    print("=" * 116)
    print(f"[*] Đã lưu kết quả đo đạc tracking tại: {csv_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá & so sánh giải thuật Tracking MOT cho YOLO26s và YOLO26m")
    parser.add_argument("--video", type=str, default="", help="Đường dẫn file video đầu vào (mặc định dataset_test/test_exp.mp4)")
    parser.add_argument("--frames", type=int, default=150, help="Số frame thực nghiệm (mặc định 150)")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame bắt đầu (mặc định 0)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "openvino", "pytorch", "cuda"], help="Backend suy luận")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tin cậy đầu vào")
    args = parser.parse_args()

    run_experiment(
        video_path=args.video,
        num_frames=args.frames,
        start_frame=args.start_frame,
        backend=args.backend,
        imgsz=args.imgsz,
        conf=args.conf,
    )
