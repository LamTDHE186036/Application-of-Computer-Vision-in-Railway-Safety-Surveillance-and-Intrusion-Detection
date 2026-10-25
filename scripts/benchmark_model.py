import os
import time
import argparse
import numpy as np
from ultralytics import YOLO

# Cho phép Ultralytics nhận diện thư mục model_openvino
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

def benchmark_model(model_path, num_frames=100, imgsz=640):
    """
    Đo đạc tốc độ suy luận (FPS & Latency) của mô hình trên thiết bị hiện tại
    """
    if not os.path.exists(model_path):
        print(f"[!] Lỗi: Không tìm thấy mô hình tại {model_path}")
        return

    print(f"[*] Nạp mô hình: {model_path}")
    model = YOLO(model_path)

    # Thử xác định device tốt nhất
    device = "cpu"
    try:
        import openvino as ov
        core = ov.Core()
        if "GPU" in core.available_devices and "openvino" in model_path.lower():
            device = "intel:gpu"
    except Exception:
        pass

    print(f"[*] Thiết bị suy luận: {device}")
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    print("[*] Đang khởi động (warmup 10 frames)...")
    for _ in range(10):
        _ = model(dummy_frame, verbose=False, device=device)

    print(f"[*] Đang thực hiện benchmark với {num_frames} frames...")
    latencies = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        _ = model(dummy_frame, verbose=False, device=device)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    fps = 1000.0 / avg_latency

    print("\n" + "=" * 60)
    print(" KẾT QUẢ BENCHMARK SUY LUẬN MÔ HÌNH")
    print(f"-> Mô hình: {model_path}")
    print(f"-> Thiết bị: {device}")
    print(f"-> Độ trễ trung bình: {avg_latency:.2f} ms")
    print(f"-> Độ trễ trung vị: {median_latency:.2f} ms")
    print(f"-> Tốc độ suy luận (FPS): {fps:.1f} FPS")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark tốc độ mô hình YOLO")
    parser.add_argument("--model", type=str, default="models/yolo26s/model_openvino",
                        help="Đường dẫn file .pt hoặc thư mục openvino")
    parser.add_argument("--frames", type=int, default=100, help="Số khung hình đo đạc")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước khung hình")
    args = parser.parse_args()

    benchmark_model(args.model, num_frames=args.frames, imgsz=args.imgsz)
