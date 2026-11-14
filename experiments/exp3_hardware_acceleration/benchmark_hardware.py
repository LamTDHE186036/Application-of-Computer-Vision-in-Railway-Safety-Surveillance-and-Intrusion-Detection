"""
Thực nghiệm 3: Benchmark Tối ưu hóa Phần cứng & Tăng tốc Suy luận (Hardware Acceleration)
Mô hình sử dụng: duy nhất yolo26s.pt và model_openvino
Dữ liệu thử nghiệm: duy nhất video Dataset_test/test_exp.mp4
So sánh hiệu năng suy luận:
1. PyTorch Framework (.pt) thực thi trên Intel CPU
2. Intel OpenVINO Toolkit (.xml/.bin) thực thi trên Intel CPU (Tối ưu AVX-512, Layer Fusion)
3. Intel OpenVINO Toolkit thực thi trên Intel Iris Xe Graphics (iGPU)
"""

import os
import time
import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Patch AutoBackend để Ultralytics nhận diện chuẩn thư mục OpenVINO
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

def resolve_pt_path(custom_path=""):
    """Tự động tìm đường dẫn mô hình PyTorch YOLO26s"""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    candidates = [
        os.path.join(BASE_DIR, "models", "yolo26s", "model_pytorch", "train-railway-yolo26s-all.pt"),
        os.path.join(BASE_DIR, "models", "yolo26s", "model_pytorch", "yolo26s.pt"),
        os.path.join(BASE_DIR, "research", "finetune", "train_yolo26s_all", "train-railway-yolo26s-all.pt"),
        os.path.join(BASE_DIR, "research", "finetune", "train_yolo26s_all", "weights", "best.pt")
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def get_hardware_info():
    """Thu thập thông tin phần cứng thiết bị Intel CPU & iGPU"""
    info = {"cpu": "Standard CPU", "gpu": None, "has_ov_gpu": False}
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        if "CPU" in devices:
            info["cpu"] = core.get_property("CPU", "FULL_DEVICE_NAME")
        if "GPU" in devices:
            info["gpu"] = core.get_property("GPU", "FULL_DEVICE_NAME")
            info["has_ov_gpu"] = True
    except Exception as e:
        info["error"] = str(e)
    return info

def load_video_frames(video_path, num_frames=150):
    """Nạp trực tiếp chuỗi khung hình thực tế từ video vào RAM"""
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
    """
    Đo đạc chi tiết độ trễ suy luận và FPS cho 1 cấu hình mô hình + thiết bị trên các khung hình thực tế
    """
    if not os.path.exists(model_path):
        print(f"[!] LỖI: Không tìm thấy mô hình tại: {model_path}")
        return None

    print(f"\n[*] Đang nạp mô hình: {os.path.basename(model_path)} (Device: {device})...")
    model = YOLO(model_path)

    warmup_frame = frames[0]

    # Warmup
    print(f"[*] Đang chạy khởi động (Warmup {warmup_iters} frames)...")
    for _ in range(warmup_iters):
        _ = model(warmup_frame, verbose=False, device=device, imgsz=imgsz)

    # Benchmark chính thức
    total_test = len(frames)
    print(f"[*] Đang thực hiện đo đạc ({total_test} frames thực tế)...")
    latencies = []
    for f in frames:
        t0 = time.perf_counter()
        _ = model(f, verbose=False, device=device, imgsz=imgsz)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

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
        "fps": round(fps, 1)
    }

def run_hardware_benchmark(pt_path=None, ov_path=None, num_frames=150, imgsz=640, video_path=None):
    # 1. Kiểm tra video Dataset_test/test_exp.mp4
    if not video_path:
        video_path = os.path.join(BASE_DIR, "Dataset_test", "test_exp.mp4")

    if not os.path.exists(video_path):
        print(f"[!] LỖI: Không tìm thấy video thực nghiệm tại: {video_path}")
        return None

    print(f"[*] Nạp video thực nghiệm: {video_path}")
    test_frames = load_video_frames(video_path, num_frames=num_frames)
    if len(test_frames) == 0:
        print(f"[!] LỖI: Không đọc được khung hình nào từ video {video_path}!")
        return None
    print(f"[+] Đã đọc thành công {len(test_frames)} frames thực tế từ video.")

    # 2. Kiểm tra mô hình YOLO26s (PyTorch và OpenVINO)
    pt_path = resolve_pt_path(pt_path)
    if not ov_path:
        ov_path = os.path.join(BASE_DIR, "models", "yolo26s", "model_openvino")

    if not os.path.exists(pt_path):
        print(f"[!] LỖI: Không tìm thấy mô hình PyTorch yolo26s.pt tại: {pt_path}")
    if not os.path.exists(ov_path):
        print(f"[!] LỖI: Không tìm thấy thư mục mô hình OpenVINO tại: {ov_path}")

    hw_info = get_hardware_info()

    print("=" * 85)
    print(" THỰC NGHIỆM 3: ĐÁNH GIÁ TĂNG TỐC PHẦN CỨNG (HARDWARE ACCELERATION)")
    print(" So sánh suy luận: PyTorch vs Intel OpenVINO trên Intel CPU & Intel iGPU")
    print("=" * 85)
    print(f"[*] Bộ vi xử lý (CPU)     : {hw_info['cpu']}")
    print(f"[*] Bộ xử lý đồ họa (iGPU): {hw_info['gpu'] or 'Không tìm thấy iGPU'}")
    print(f"[*] Kích thước khung hình : {imgsz}x{imgsz}")
    print(f"[*] Số khung hình đo đạc  : {len(test_frames)} frames/cấu hình")
    print("=" * 85)

    results = {}

    # 1. Đo lường PyTorch CPU
    if os.path.exists(pt_path):
        print("\n>>> [1/3] Đo lường PyTorch Baseline trên CPU...")
        results["pytorch_cpu"] = benchmark_single_target(pt_path, frames=test_frames, device="cpu", imgsz=imgsz)
    else:
        print(f"[!] Bỏ qua PyTorch CPU (Không tìm thấy: {pt_path})")

    # 2. Đo lường OpenVINO CPU
    if os.path.exists(ov_path):
        print("\n>>> [2/3] Đo lường Intel OpenVINO trên CPU (Tối ưu AVX-512 & Layer Fusion)...")
        results["openvino_cpu"] = benchmark_single_target(ov_path, frames=test_frames, device="cpu", imgsz=imgsz)
    else:
        print(f"[!] Bỏ qua OpenVINO CPU (Không tìm thấy: {ov_path})")

    # 3. Đo lường OpenVINO iGPU
    if os.path.exists(ov_path) and hw_info["has_ov_gpu"]:
        print("\n>>> [3/3] Đo lường Intel OpenVINO trên Intel Iris Xe iGPU...")
        try:
            results["openvino_gpu"] = benchmark_single_target(ov_path, frames=test_frames, device="intel:gpu", imgsz=imgsz)
        except Exception as e:
            print(f"[!] Lỗi khi đo trên intel:gpu ({e}), đang thử lại với device='GPU'...")
            try:
                results["openvino_gpu"] = benchmark_single_target(ov_path, frames=test_frames, device="GPU", imgsz=imgsz)
            except Exception as e2:
                print(f"[!] Thử lại với device='GPU' thất bại: {e2}")

    # Bảng tổng kết
    baseline_lat = results.get("pytorch_cpu", {}).get("avg_lat", 0.0) if "pytorch_cpu" in results else 0.0

    print("\n\n" + "=" * 102)
    print(" BẢNG TỔNG HỢP HIỆU NĂNG TĂNG TỐC PHẦN CỨNG (INFERENCE BENCHMARK)")
    print("=" * 102)
    print(f"{'Cấu hình Thực thi':<28} | {'Thiết bị':<20} | {'Avg (ms)':<10} | {'P95 (ms)':<10} | {'FPS':<8} | {'Tăng tốc':<10}")
    print("-" * 102)

    rows = []
    configs = [
        ("pytorch_cpu", "1. PyTorch Framework (.pt)", "Intel CPU"),
        ("openvino_cpu", "2. Intel OpenVINO (.xml)", "Intel CPU"),
        ("openvino_gpu", "3. Intel OpenVINO (.xml)", "Intel Iris Xe iGPU")
    ]

    for key, label, dev_name in configs:
        if key in results and results[key]:
            r = results[key]
            if key == "pytorch_cpu" or baseline_lat <= 0:
                speedup_str = "1.00x (Gốc)"
            else:
                speedup = baseline_lat / r["avg_lat"] if r["avg_lat"] > 0 else 0.0
                speedup_str = f"{speedup:.2f}x"

            print(f"{label:<28} | {dev_name:<20} | {r['avg_lat']:<10} | {r['p95_lat']:<10} | {r['fps']:<8} | {speedup_str:<10}")

            rows.append({
                "Configuration": label[3:],
                "Device": dev_name,
                "Avg_Latency_ms": r["avg_lat"],
                "Median_Latency_ms": r["med_lat"],
                "P95_Latency_ms": r["p95_lat"],
                "Min_Latency_ms": r["min_lat"],
                "Max_Latency_ms": r["max_lat"],
                "FPS": r["fps"],
                "Speedup": speedup_str
            })

    print("=" * 102)

    # Tự động lưu bảng kết quả vào thư mục results/
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "hardware_benchmark.csv")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"[*] Đã lưu kết quả đo đạc phần cứng đầy đủ tại: {csv_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đo đạc tăng tốc phần cứng PyTorch vs OpenVINO")
    parser.add_argument("--pt", type=str, default="", help="Đường dẫn file .pt của YOLO26s")
    parser.add_argument("--ov", type=str, default="", help="Đường dẫn thư mục OpenVINO của YOLO26s")
    parser.add_argument("--video", type=str, default="", help="Đường dẫn file video thực nghiệm (mặc định Dataset_test/test_exp.mp4)")
    parser.add_argument("--frames", type=int, default=150, help="Số khung hình đo (mặc định 150)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    args = parser.parse_args()

    run_hardware_benchmark(pt_path=args.pt, ov_path=args.ov, num_frames=args.frames, imgsz=args.imgsz, video_path=args.video)