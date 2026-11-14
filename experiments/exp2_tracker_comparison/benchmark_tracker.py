"""
Thực nghiệm 2: Đánh giá & So sánh Giải thuật Tracking đa đối tượng (MOT)
Mô hình sử dụng: duy nhất yolo26s.pt
Dữ liệu thử nghiệm: duy nhất video Dataset_test/test_exp.mp4
So sánh 3 phương pháp:
1. No Tracking (Chỉ sử dụng YOLO Detection thuần)
2. BoT-SORT / DeepSORT (Tracking kết hợp Re-ID ngoại hình & Kalman Filter)
3. ByteTrack (Tracking liên kết hai tầng điểm tin cậy dựa trên IoU & Kalman Filter - Đề xuất)
"""

import os
import time
import argparse
import numpy as np
import cv2
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def resolve_yolo26_path(custom_path=""):
    """Tự động tìm đường dẫn mô hình YOLO26s hợp lệ"""
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

def evaluate_tracking_mode(model_path, mode, frames, tracker_cfg=None, warmup_iters=10, conf=0.25):
    """
    Đo đạc hiệu năng và chất lượng duy trì ID của một chế độ tracking trên chuỗi khung hình thực tế.
    Có bước Warmup riêng để tránh nhiễu khởi tạo ở frame 0.
    """
    model = YOLO(model_path)
    cfg = tracker_cfg if (tracker_cfg and os.path.exists(tracker_cfg)) else "bytetrack.yaml"

    # 1. Bước khởi động (Warmup) để nạp bộ nhớ và biên dịch đồ thị
    warmup_frame = frames[0]
    for _ in range(warmup_iters):
        if mode == "none":
            _ = model.predict(warmup_frame, verbose=False, conf=conf)
        elif mode == "botsort":
            _ = model.track(warmup_frame, tracker="botsort.yaml", verbose=False, conf=conf, persist=True)
        elif mode == "bytetrack":
            _ = model.track(warmup_frame, tracker=cfg, verbose=False, conf=conf, persist=True)

    # Khởi tạo lại model/tracker sạch sẽ trước khi bước vào vòng lặp đo chính thức
    model = YOLO(model_path)

    latencies = []
    unique_ids = set()
    id_frame_history = defaultdict(list)

    # 2. Vòng lặp đo đạc chính thức
    for f_idx, frame in enumerate(frames):
        t0 = time.perf_counter()

        if mode == "none":
            results = model.predict(frame, verbose=False, conf=conf)
        elif mode == "botsort":
            results = model.track(frame, tracker="botsort.yaml", verbose=False, conf=conf, persist=True)
        elif mode == "bytetrack":
            results = model.track(frame, tracker=cfg, verbose=False, conf=conf, persist=True)

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_elapsed)

        # Thu thập thống kê lịch sử xuất hiện của từng ID
        if mode != "none" and results[0].boxes.id is not None:
            current_ids = results[0].boxes.id.int().cpu().tolist()
            for tid in current_ids:
                unique_ids.add(tid)
                id_frame_history[tid].append(f_idx)

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # 3. Tính toán chỉ số định lượng chất lượng duy trì ID (Dwell Time & ID Stability)
    if mode == "none" or len(unique_ids) == 0:
        avg_lifespan_frames = 0.0
        fragmentations = 0
        dwell_status = "Không thể đo (Không có ID)" if mode == "none" else "Không phát hiện đối tượng"
    else:
        lifespans = [ len(f_list) for f_list in id_frame_history.values() ]
        avg_lifespan_frames = float(np.mean(lifespans))
        
        # Đếm số lần một ID bị ngắt quãng giữa chừng rồi xuất hiện lại (ID Fragmentation)
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
        "dwell_support": dwell_status
    }

def run_experiment(video_path=None, model_path=None, num_frames=150):
    print("=" * 80)
    print(" THỰC NGHIỆM 2: SO SÁNH GIẢI THUẬT TRACKING ĐA ĐỐI TƯỢNG (MOT)")
    print(" Hệ thống Giám sát An toàn Đường sắt & Phát hiện Xâm nhập")
    print("=" * 80)

    # 1. Kiểm tra mô hình YOLO26s
    model_path = resolve_yolo26_path(model_path)
    if not os.path.exists(model_path):
        print(f"[!] LỖI: Không tìm thấy mô hình yolo26s.pt tại: {model_path}")
        return None
    print(f"[*] Mô hình sử dụng       : {model_path}")

    # 2. Kiểm tra video Dataset_test/test_exp.mp4
    if not video_path:
        video_path = os.path.join(BASE_DIR, "Dataset_test", "test_exp.mp4")

    if not os.path.exists(video_path):
        print(f"[!] LỖI: Không tìm thấy video thực nghiệm tại: {video_path}")
        return None

    print(f"[*] Nạp video thực nghiệm : {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        print(f"[!] LỖI: Không đọc được khung hình nào từ video {video_path}!")
        return None
    print(f"[+] Đã nạp thành công {len(frames)} khung hình vào RAM để loại bỏ độ trễ đọc đĩa.")

    # 3. Chạy đánh giá qua 3 giải thuật
    tracker_cfg = os.path.join(os.path.dirname(__file__), "bytetrack.yaml")

    print("\n[1/3] Đang đánh giá chế độ: Không Tracking (Pure Detection)...")
    res_none = evaluate_tracking_mode(model_path, "none", frames)

    print("[2/3] Đang đánh giá chế độ: BoT-SORT / DeepSORT (Appearance Re-ID + Motion)...")
    res_botsort = evaluate_tracking_mode(model_path, "botsort", frames)

    print("[3/3] Đang đánh giá chế độ: ByteTrack (Two-Stage IoU Matching - Đề xuất)...")
    res_bytetrack = evaluate_tracking_mode(model_path, "bytetrack", frames, tracker_cfg=tracker_cfg)

    # Tính toán chính xác độ trễ tăng thêm (Overhead) so với Pure Detection
    base_lat = res_none["avg_latency_ms"]
    res_none["overhead_ms"] = 0.0
    res_botsort["overhead_ms"] = round(res_botsort["avg_latency_ms"] - base_lat, 2)
    res_bytetrack["overhead_ms"] = round(res_bytetrack["avg_latency_ms"] - base_lat, 2)

    # 4. Hiển thị bảng tổng kết
    print("\n" + "=" * 105)
    print(" BẢNG TỔNG KẾT KẾT QUẢ THỰC NGHIỆM TRACKING ĐA ĐỐI TƯỢNG")
    print("=" * 105)
    print(f"{'Giải thuật':<22} | {'Độ trễ (ms)':<12} | {'Overhead (ms)':<14} | {'Tốc độ (FPS)':<13} | {'Số ID':<7} | {'Đánh giá Dwell Time':<28}")
    print("-" * 105)
    for label, r in [("1. Pure YOLO", res_none), ("2. BoT-SORT/DeepSORT", res_botsort), ("3. ByteTrack", res_bytetrack)]:
        ov_str = f"+{r['overhead_ms']:.2f}" if r['overhead_ms'] >= 0 else f"{r['overhead_ms']:.2f}"
        print(f"{label:<22} | {r['avg_latency_ms']:<12} | {ov_str:<14} | {r['fps']:<13} | {str(r['total_unique_ids']):<7} | {r['dwell_support']:<28}")
    print("=" * 105)

    # 5. Tự động lưu bảng kết quả vào thư mục results/
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "tracker_comparison.csv")

    df = pd.DataFrame([
        {
            "Method": "Pure YOLO (No Tracking)",
            "Latency_ms": res_none["avg_latency_ms"],
            "P95_Latency_ms": res_none["p95_latency_ms"],
            "Overhead_ms": res_none["overhead_ms"],
            "FPS": res_none["fps"],
            "Unique_IDs": res_none["total_unique_ids"],
            "Avg_Lifespan_Frames": res_none["avg_lifespan_frames"],
            "ID_Fragmentations": res_none["id_fragmentations"],
            "Dwell_Support": res_none["dwell_support"]
        },
        {
            "Method": "BoT-SORT / DeepSORT",
            "Latency_ms": res_botsort["avg_latency_ms"],
            "P95_Latency_ms": res_botsort["p95_latency_ms"],
            "Overhead_ms": res_botsort["overhead_ms"],
            "FPS": res_botsort["fps"],
            "Unique_IDs": res_botsort["total_unique_ids"],
            "Avg_Lifespan_Frames": res_botsort["avg_lifespan_frames"],
            "ID_Fragmentations": res_botsort["id_fragmentations"],
            "Dwell_Support": res_botsort["dwell_support"]
        },
        {
            "Method": "ByteTrack (Đề xuất)",
            "Latency_ms": res_bytetrack["avg_latency_ms"],
            "P95_Latency_ms": res_bytetrack["p95_latency_ms"],
            "Overhead_ms": res_bytetrack["overhead_ms"],
            "FPS": res_bytetrack["fps"],
            "Unique_IDs": res_bytetrack["total_unique_ids"],
            "Avg_Lifespan_Frames": res_bytetrack["avg_lifespan_frames"],
            "ID_Fragmentations": res_bytetrack["id_fragmentations"],
            "Dwell_Support": res_bytetrack["dwell_support"]
        }
    ])
    df.to_csv(csv_path, index=False)
    print(f"[*] Đã lưu kết quả đo đạc tracking tại: {csv_path}")

    return res_none, res_botsort, res_bytetrack

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá & so sánh giải thuật Tracking MOT")
    parser.add_argument("--video", type=str, default="", help="Đường dẫn file video đầu vào (mặc định Dataset_test/test_exp.mp4)")
    parser.add_argument("--model", type=str, default="", help="Đường dẫn mô hình YOLO26s")
    parser.add_argument("--frames", type=int, default=150, help="Số frame thực nghiệm (mặc định 150)")
    args = parser.parse_args()

    run_experiment(video_path=args.video, model_path=args.model, num_frames=args.frames)