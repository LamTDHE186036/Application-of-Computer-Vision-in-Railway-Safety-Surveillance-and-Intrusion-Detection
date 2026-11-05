"""
Thực nghiệm 3: Benchmark Tối ưu hóa Phần cứng & Tăng tốc Suy luận (Hardware Acceleration)
So sánh hiệu năng suy luận mô hình nhận diện an toàn đường sắt YOLO:
1. PyTorch Framework (.pt) thực thi trên Intel CPU
2. Intel OpenVINO Toolkit (.xml/.bin) thực thi trên Intel CPU (Tối ưu AVX-512, Layer Fusion)
3. Intel OpenVINO Toolkit thực thi trên Intel Iris Xe Graphics (iGPU)
"""

import os
import time
import argparse
import numpy as np
from ultralytics import YOLO

# Patch AutoBackend để Ultralytics nhận diện chuẩn thư mục OpenVINO
try:
    from ultralytics.nn import autobackend
    _orig_model_type = autobackend.AutoBackend._model_type

    def _patched_model_type(p, dnn=False):
        fmt = _orig_model_type(p, dnn)
        if fmt is None and 'openvino' in str(p).lower():
            return 'openvino'
        return fmt

    autobackend.AutoBackend._model_type = staticmethod(_patched_model_type)
except Exception:
    pass

def get_hardware_info():
    """
    Thu thập thông tin phần cứng thiết bị Intel CPU & iGPU
    """
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

def benchmark_single_target(model_path, device="cpu", num_frames=60, imgsz=640):
    """
    Đo đạc chi tiết độ trễ suy luận và FPS cho 1 cấu hình mô hình + thiết bị
    """
    if not os.path.exists(model_path):
        return None

    print(f"\n[*] Đang nạp mô hình: {os.path.basename(model_path)} (Device: {device})...")
    model = YOLO(model_path)

    # Sinh frame giả lập ngẫu nhiên
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    print(f"[*] Đang chạy khởi động (Warmup 10 frames)...")
    for _ in range(10):
        _ = model(dummy_frame, verbose=False, device=device)

    # Benchmark chính thức
    print(f"[*] Đang thực hiện đo đạc ({num_frames} frames)...")
    latencies = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        _ = model(dummy_frame, verbose=False, device=device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000) # milliseconds

    avg_lat = float(np.mean(latencies))
    med_lat = float(np.median(latencies))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    return {
        "avg_lat": round(avg_lat, 2),
        "med_lat": round(med_lat, 2),
        "min_lat": round(min_lat, 2),
        "max_lat": round(max_lat, 2),
        "p95_lat": round(p95_lat, 2),
        "fps": round(fps, 1)
    }

def run_hardware_benchmark(pt_path=None, ov_path=None, num_frames=60, imgsz=640):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Thiết lập đường dẫn mặc định nếu không truyền (Lấy mô hình hoàn thiện train_yolo26s_all)
    if not pt_path:
        candidates = [
            os.path.join(base_dir, "research", "finetune", "train_yolo26s_all", "train-railway-yolo26s-all.pt"),
            os.path.join(base_dir, "models", "yolo26s", "model_pytorch", "train-railway-yolo26s-all.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                pt_path = c
                break
    if not ov_path:
        ov_path = os.path.join(base_dir, "models", "yolo26s", "model_openvino")

    hw_info = get_hardware_info()

    print("=" * 80)
    print(" THỰC NGHIỆM 3: ĐÁNH GIÁ TĂNG TỐC PHẦN CỨNG (HARDWARE ACCELERATION)")
    print(" So sánh suy luận: PyTorch vs Intel OpenVINO trên Intel CPU & Intel iGPU")
    print("=" * 80)
    print(f"[*] Bộ vi xử lý (CPU)     : {hw_info['cpu']}")
    print(f"[*] Bộ xử lý đồ họa (iGPU): {hw_info['gpu'] or 'Không tìm thấy iGPU'}")
    print(f"[*] Kích thước khung hình : {imgsz}x{imgsz}")
    print(f"[*] Số khung hình đo đạc  : {num_frames} frames/cấu hình")
    print("=" * 80)

    results = {}

    # 1. Đo lường PyTorch CPU
    if os.path.exists(pt_path):
        print("\n>>> [1/3] Đo lường PyTorch Baseline trên CPU...")
        results["pytorch_cpu"] = benchmark_single_target(pt_path, device="cpu", num_frames=num_frames, imgsz=imgsz)
    else:
        print(f"[!] Bỏ qua PyTorch CPU (Không tìm thấy: {pt_path})")

    # 2. Đo lường OpenVINO CPU
    if os.path.exists(ov_path):
        print("\n>>> [2/3] Đo lường Intel OpenVINO trên CPU (Tối ưu AVX-512 & Layer Fusion)...")
        results["openvino_cpu"] = benchmark_single_target(ov_path, device="cpu", num_frames=num_frames, imgsz=imgsz)
    else:
        print(f"[!] Bỏ qua OpenVINO CPU (Không tìm thấy: {ov_path})")

    # 3. Đo lường OpenVINO iGPU
    if os.path.exists(ov_path) and hw_info["has_ov_gpu"]:
        print("\n>>> [3/3] Đo lường Intel OpenVINO trên Intel Iris Xe iGPU...")
        try:
            results["openvino_gpu"] = benchmark_single_target(ov_path, device="intel:gpu", num_frames=num_frames, imgsz=imgsz)
        except Exception as e:
            print(f"[!] Lỗi khi đo trên GPU: {e}")
            # Thử device=GPU
            try:
                results["openvino_gpu"] = benchmark_single_target(ov_path, device="GPU", num_frames=num_frames, imgsz=imgsz)
            except Exception as e2:
                print(f"[!] Thử lại với device='GPU' thất bại: {e2}")

    # Bảng tổng kết
    baseline_lat = results.get("pytorch_cpu", {}).get("avg_lat", 1.0) if "pytorch_cpu" in results else 1.0

    print("\n\n" + "=" * 90)
    print(" BẢNG TỔNG HỢP HIỆU NĂNG TĂNG TỐC PHẦN CỨNG (INFERENCE BENCHMARK)")
    print("=" * 90)
    print(f"{'Cấu hình Thực thi':<28} | {'Thiết bị':<22} | {'Latency (ms)':<12} | {'FPS':<10} | {'Tăng tốc':<10}")
    print("-" * 90)

    if "pytorch_cpu" in results and results["pytorch_cpu"]:
        r = results["pytorch_cpu"]
        print(f"{'1. PyTorch Framework (.pt)':<28} | {'Intel CPU':<22} | {r['avg_lat']:<12} | {r['fps']:<10} | {'1.00x (Gốc)':<10}")

    if "openvino_cpu" in results and results["openvino_cpu"]:
        r = results["openvino_cpu"]
        speedup = baseline_lat / r['avg_lat'] if r['avg_lat'] > 0 else 0
        print(f"{'2. Intel OpenVINO (.xml)':<28} | {'Intel CPU':<22} | {r['avg_lat']:<12} | {r['fps']:<10} | {f'{speedup:.2f}x':<10}")

    if "openvino_gpu" in results and results["openvino_gpu"]:
        r = results["openvino_gpu"]
        speedup = baseline_lat / r['avg_lat'] if r['avg_lat'] > 0 else 0
        print(f"{'3. Intel OpenVINO (.xml)':<28} | {'Intel Iris Xe iGPU':<22} | {r['avg_lat']:<12} | {r['fps']:<10} | {f'{speedup:.2f}x':<10}")

    print("=" * 90)

    # 4. Tự động lưu bảng kết quả vào thư mục results/
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "hardware_benchmark.csv")

    import pandas as pd
    rows = []
    if "pytorch_cpu" in results and results["pytorch_cpu"]:
        r = results["pytorch_cpu"]
        rows.append({"Configuration": "PyTorch Framework (.pt)", "Device": "Intel CPU", "Latency_ms": r["avg_lat"], "Median_ms": r["med_lat"], "FPS": r["fps"], "Speedup": "1.00x"})
    if "openvino_cpu" in results and results["openvino_cpu"]:
        r = results["openvino_cpu"]
        speedup = baseline_lat / r['avg_lat'] if r['avg_lat'] > 0 else 0
        rows.append({"Configuration": "Intel OpenVINO (.xml)", "Device": "Intel CPU", "Latency_ms": r["avg_lat"], "Median_ms": r["med_lat"], "FPS": r["fps"], "Speedup": f"{speedup:.2f}x"})
    if "openvino_gpu" in results and results["openvino_gpu"]:
        r = results["openvino_gpu"]
        speedup = baseline_lat / r['avg_lat'] if r['avg_lat'] > 0 else 0
        rows.append({"Configuration": "Intel OpenVINO (.xml)", "Device": "Intel Iris Xe iGPU", "Latency_ms": r["avg_lat"], "Median_ms": r["med_lat"], "FPS": r["fps"], "Speedup": f"{speedup:.2f}x"})

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"[*] Đã lưu kết quả đo đạc phần cứng tại: {csv_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đo đạc tăng tốc phần cứng PyTorch vs OpenVINO")
    parser.add_argument("--pt", type=str, default="", help="Đường dẫn file .pt")
    parser.add_argument("--ov", type=str, default="", help="Đường dẫn thư mục OpenVINO")
    parser.add_argument("--frames", type=int, default=50, help="Số khung hình đo")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    args = parser.parse_args()

    run_hardware_benchmark(pt_path=args.pt, ov_path=args.ov, num_frames=args.frames, imgsz=args.imgsz)

