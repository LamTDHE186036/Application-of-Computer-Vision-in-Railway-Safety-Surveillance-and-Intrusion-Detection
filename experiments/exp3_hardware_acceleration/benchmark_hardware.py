"""
Thực nghiệm 3: Benchmark Tối ưu hóa Phần cứng & Tăng tốc Suy luận (Hardware Acceleration)
Đánh giá song song cả 2 mô hình của hệ thống:
1. YOLO26s (PyTorch .pt & OpenVINO FP16 .xml/.bin)
2. YOLO26m (PyTorch .pt & OpenVINO FP16 .xml/.bin)

Dữ liệu thử nghiệm: dataset_test/test_exp.mp4 (Warmup 10 frames + đo trên N frames thực tế)
So sánh 3 cấu hình cho mỗi mô hình:
1. PyTorch Framework (.pt) thực thi trên Intel CPU (Baseline)
2. Intel OpenVINO FP16 (.xml/.bin) thực thi trên Intel CPU
3. Intel OpenVINO FP16 (.xml/.bin) thực thi trên Intel Iris Xe Graphics (iGPU)
"""

import os
import sys
import time
import tempfile
import argparse
import cv2
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EXP3_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(EXP3_DIR, "..", ".."))
RESULTS_DIR = os.path.join(EXP3_DIR, "results")

if not os.environ.get("YOLO_CONFIG_DIR"):
    _tmp_cfg = os.path.join(tempfile.gettempdir(), "railsafe_ultralytics_cfg")
    try:
        os.makedirs(_tmp_cfg, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = _tmp_cfg
    except Exception:
        _local_cfg = os.path.join(EXP3_DIR, ".ultralytics_cfg")
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


def resolve_model_pairs():
    return {
        "YOLO26s": {
            "pt": first_existing([
                os.path.join(BASE_DIR, "models", "yolo26s", "model_pytorch", "train-railway-yolo26s-all.pt"),
                os.path.join(BASE_DIR, "models", "yolo26s", "model_pytorch", "yolo26s.pt"),
                os.path.join(BASE_DIR, "research", "finetune", "train_yolo26s", "yolo26s.pt"),
            ]),
            "ov": os.path.join(BASE_DIR, "models", "yolo26s", "model_openvino"),
        },
        "YOLO26m": {
            "pt": first_existing([
                os.path.join(BASE_DIR, "models", "yolo26m", "model_pytorch", "yolo26m.pt"),
                os.path.join(BASE_DIR, "research", "finetune", "train_yolo26m", "yolo26m.pt"),
                os.path.join(BASE_DIR, "Fine-tune-Dataset-Lam", "Fine-tune-Dataset-Lam", "model", "best.pt"),
            ]),
            "ov": os.path.join(BASE_DIR, "models", "yolo26m", "model_openvino"),
        },
    }


def get_hardware_info():
    """Thu thập thông tin phần cứng thiết bị Intel CPU & iGPU"""
    info = {"cpu": "Standard CPU", "gpu": None, "has_ov_gpu": False, "has_cuda": False, "cuda_name": None}
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["has_cuda"] = True
            info["cuda_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        if "CPU" in devices:
            info["cpu"] = core.get_property("CPU", "FULL_DEVICE_NAME")
        if "GPU" in devices:
            info["gpu"] = core.get_property("GPU", "FULL_DEVICE_NAME")
            info["has_ov_gpu"] = True
    except Exception:
        pass
    return info


def load_video_frames(video_path, num_frames=50):
    frames = []
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
        cap.release()
    return frames


def benchmark_single_target(model_path, frames, device="cpu", imgsz=640, warmup_iters=10):
    if not os.path.exists(model_path):
        print(f"[!] LỖI: Không tìm thấy mô hình tại: {model_path}")
        return None

    print(f"    [*] Nạp {os.path.basename(model_path)} (Device: {device})...")
    model = YOLO(model_path)

    warmup_frame = frames[0]
    for _ in range(warmup_iters):
        _ = model(warmup_frame, verbose=False, device=device, imgsz=imgsz)

    latencies = []
    for f in frames:
        t0 = time.perf_counter()
        _ = model(f, verbose=False, device=device, imgsz=imgsz)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    med_lat = float(np.median(latencies)) if latencies else 0.0
    min_lat = float(np.min(latencies)) if latencies else 0.0
    max_lat = float(np.max(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    return {
        "avg_lat": round(avg_lat, 2),
        "med_lat": round(med_lat, 2),
        "min_lat": round(min_lat, 2),
        "max_lat": round(max_lat, 2),
        "p95_lat": round(p95_lat, 2),
        "fps": round(fps, 1),
    }


def generate_chart(df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    models = df["Model"].unique().tolist()
    configs = [
        ("PyTorch (.pt) — Intel CPU", "#64748b"),
        ("OpenVINO FP16 — Intel CPU", "#f59e0b"),
        ("OpenVINO FP16 — Intel Iris Xe iGPU", "#0d828a"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(models))
    width = 0.25

    for idx, (cfg_label, col) in enumerate(configs):
        fps_vals, lat_vals = [], []
        for mod in models:
            sub = df[(df["Model"] == mod) & (df["Config_Label"] == cfg_label)]
            fps_vals.append(sub.iloc[0]["FPS"] if not sub.empty else 0.0)
            lat_vals.append(sub.iloc[0]["Avg_Latency_ms"] if not sub.empty else 0.0)

        offset = (idx - 1) * width
        b1 = axes[0].bar(x + offset, fps_vals, width, label=cfg_label, color=col)
        for b in b1:
            h = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)

        b2 = axes[1].bar(x + offset, lat_vals, width, label=cfg_label, color=col)
        for b in b2:
            h = b.get_height()
            axes[1].text(b.get_x() + b.get_width() / 2, h + 6, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    axes[0].axhline(25.0, color="#ef4444", linestyle="--", linewidth=1.3, label="Chuẩn Real-Time Camera (25 FPS)")
    axes[0].set_title("Tốc độ suy luận (FPS — cao hơn tốt hơn)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("FPS")
    axes[0].set_ylim(0, max(df["FPS"].max() * 1.25, 38))
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Độ trễ trung bình (ms/frame — thấp hơn tốt hơn)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Avg Latency (ms)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "chart_hardware_benchmark.png"), dpi=160)
    plt.close(fig)


def run_hardware_benchmark(num_frames=50, imgsz=640, video_path=None):
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

    test_frames = load_video_frames(video_path, num_frames=num_frames)
    if not test_frames:
        print(f"[!] LỖI: Không đọc được khung hình nào từ video {video_path}!")
        return None

    hw_info = get_hardware_info()
    model_pairs = resolve_model_pairs()

    print("=" * 100)
    print(" THỰC NGHIỆM 3: ĐÁNH GIÁ TĂNG TỐC PHẦN CỨNG (PyTorch vs Intel OpenVINO) — YOLO26s & YOLO26m")
    print("=" * 100)
    print(f"[*] Bộ vi xử lý (CPU)     : {hw_info['cpu']}")
    print(f"[*] Đồ họa tích hợp (iGPU): {hw_info['gpu'] or 'Không tìm thấy'}")
    print(f"[*] GPU rời (NVIDIA CUDA) : {hw_info['cuda_name'] or 'Không có'}")
    print(f"[*] Video & Khung hình    : {video_path} ({len(test_frames)} frames, warmup=10, imgsz={imgsz}x{imgsz})")
    print("=" * 100)

    rows = []

    for model_name, paths in model_pairs.items():
        pt_path = paths["pt"]
        ov_path = paths["ov"]
        print(f"\n>>> Đang benchmark phần cứng cho {model_name}:")
        model_results = {}

        if os.path.exists(pt_path):
            model_results["pytorch_cpu"] = benchmark_single_target(pt_path, test_frames, device="cpu", imgsz=imgsz)
        else:
            print(f"    [!] Thiếu PyTorch weights: {pt_path}")

        if os.path.isdir(ov_path):
            model_results["openvino_cpu"] = benchmark_single_target(ov_path, test_frames, device="cpu", imgsz=imgsz)
            if hw_info["has_ov_gpu"]:
                try:
                    model_results["openvino_gpu"] = benchmark_single_target(ov_path, test_frames, device="intel:gpu", imgsz=imgsz)
                except Exception as e:
                    print(f"    [!] Lỗi intel:gpu ({e}), thử device='GPU'...")
                    try:
                        model_results["openvino_gpu"] = benchmark_single_target(ov_path, test_frames, device="GPU", imgsz=imgsz)
                    except Exception as e2:
                        print(f"    [!] Thử device='GPU' thất bại: {e2}")
        else:
            print(f"    [!] Thiếu thư mục OpenVINO: {ov_path}")

        if hw_info["has_cuda"] and os.path.exists(pt_path):
            model_results["pytorch_cuda"] = benchmark_single_target(pt_path, test_frames, device="cuda", imgsz=imgsz)

        baseline_lat = (model_results.get("pytorch_cpu") or {}).get("avg_lat", 0.0)
        cfg_map = [
            ("pytorch_cpu", "PyTorch Framework (.pt)", "Intel CPU", "PyTorch (.pt) — Intel CPU"),
            ("openvino_cpu", "Intel OpenVINO FP16 (.xml)", "Intel CPU", "OpenVINO FP16 — Intel CPU"),
            ("openvino_gpu", "Intel OpenVINO FP16 (.xml)", "Intel Iris Xe iGPU", "OpenVINO FP16 — Intel Iris Xe iGPU"),
            ("pytorch_cuda", "PyTorch Framework (.pt)", hw_info["cuda_name"] or "NVIDIA CUDA", "PyTorch (.pt) — NVIDIA CUDA"),
        ]

        for key, cfg_name, dev_name, cfg_label in cfg_map:
            r = model_results.get(key)
            if not r:
                continue
            if key == "pytorch_cpu" or baseline_lat <= 0:
                speedup_str = "1.00x (Gốc)"
            else:
                speedup_str = f"{baseline_lat / r['avg_lat']:.2f}x"

            rows.append({
                "Model": model_name,
                "Configuration": cfg_name,
                "Device": dev_name,
                "Config_Label": cfg_label,
                "Frames": len(test_frames),
                "Avg_Latency_ms": r["avg_lat"],
                "Median_Latency_ms": r["med_lat"],
                "P95_Latency_ms": r["p95_lat"],
                "Min_Latency_ms": r["min_lat"],
                "Max_Latency_ms": r["max_lat"],
                "FPS": r["fps"],
                "Speedup": speedup_str,
                "RealTime_25FPS": "Đạt" if r["fps"] >= 25.0 else "Không đạt",
            })
            print(
                f"    [+] {model_name} | {cfg_label:<34} -> Avg={r['avg_lat']}ms | "
                f"P95={r['p95_lat']}ms | FPS={r['fps']} | Speedup={speedup_str}"
            )

    if not rows:
        return None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "hardware_benchmark.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    generate_chart(df)

    print("\n" + "=" * 116)
    print(" BẢNG TỔNG HỢP HIỆU NĂNG TĂNG TỐC PHẦN CỨNG — YOLO26s & YOLO26m")
    print("=" * 116)
    cols_show = ["Model", "Configuration", "Device", "Avg_Latency_ms", "Median_Latency_ms", "P95_Latency_ms", "FPS", "Speedup", "RealTime_25FPS"]
    print(df[cols_show].to_string(index=False))
    print("=" * 116)
    print(f"[*] Đã lưu kết quả đo đạc phần cứng tại: {csv_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đo đạc tăng tốc phần cứng PyTorch vs OpenVINO cho YOLO26s & YOLO26m")
    parser.add_argument("--video", type=str, default="", help="Đường dẫn file video thực nghiệm (mặc định dataset_test/test_exp.mp4)")
    parser.add_argument("--frames", type=int, default=50, help="Số khung hình đo (mặc định 50, sau 10 warmup)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    args = parser.parse_args()

    run_hardware_benchmark(num_frames=args.frames, imgsz=args.imgsz, video_path=args.video)
