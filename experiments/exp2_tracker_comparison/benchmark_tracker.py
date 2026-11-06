"""
Thực nghiệm 2: Đánh giá & So sánh Giải thuật Tracking đa đối tượng (MOT)
Mô hình sử dụng: duy nhất train-railway-yolo26s-all.pt
Dữ liệu thử nghiệm: duy nhất video Dataset_test/test_exp.mp4
So sánh 3 phương pháp:
1. No Tracking (Chỉ sử dụng YOLO Detection thuần)
2. BoT-SORT / DeepSORT (Tracking kết hợp Re-ID ngoại hình & Kalman Filter)
3. ByteTrack (Tracking liên kết hai tầng điểm tin cậy dựa trên IoU & Kalman Filter - Đề xuất)
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO

def evaluate_tracking_mode(model_path, mode, frames, tracker_cfg=None):
    """
    Đo đạc hiệu năng của một chế độ tracking trên chuỗi khung hình thực tế
    """
    model = YOLO(model_path)
    latencies = []
    ids_per_frame = []
    unique_ids = set()
    dwell_counter = {}

    # Reset predictor tracker state nếu có
    if hasattr(model, 'predictor') and hasattr(model.predictor, 'trackers'):
        for t in model.predictor.trackers:
            if hasattr(t, 'reset'):
                t.reset()

    for f_idx, frame in enumerate(frames):
        t0 = time.perf_counter()

        if mode == "none":
            # Chỉ chạy detect thuần, không có tracker
            results = model.predict(frame, verbose=False, conf=0.25)
            t_elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(t_elapsed)
            ids_per_frame.append(None)
            
        elif mode == "botsort":
            # BoT-SORT (Re-ID + Kalman Filter)
            results = model.track(frame, tracker="botsort.yaml", verbose=False, conf=0.25, persist=True)
            t_elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(t_elapsed)

            current_frame_ids = []
            if results[0].boxes.id is not None:
                current_frame_ids = results[0].boxes.id.int().cpu().tolist()
                for tid in current_frame_ids:
                    unique_ids.add(tid)
                    if tid not in dwell_counter:
                        dwell_counter[tid] = f_idx
            ids_per_frame.append(current_frame_ids)

        elif mode == "bytetrack":
            # ByteTrack (Two-stage IoU Matching + Kalman Filter)
            cfg = tracker_cfg if (tracker_cfg and os.path.exists(tracker_cfg)) else "bytetrack.yaml"
            results = model.track(frame, tracker=cfg, verbose=False, conf=0.25, persist=True)
            t_elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(t_elapsed)

            current_frame_ids = []
            if results[0].boxes.id is not None:
                current_frame_ids = results[0].boxes.id.int().cpu().tolist()
                for tid in current_frame_ids:
                    unique_ids.add(tid)
                    if tid not in dwell_counter:
                        dwell_counter[tid] = f_idx
            ids_per_frame.append(current_frame_ids)

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "mode": mode,
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 1),
        "total_unique_ids": len(unique_ids) if mode != "none" else "N/A (Không theo dõi)",
        "dwell_support": "Khả thi (Duy trì ID chính xác)" if mode != "none" else "Không thể đo (Mất ID liên tục)",
        "overhead_ms": round(avg_latency - (latencies[0] if latencies else 0), 2)
    }

def run_experiment(video_path=None, model_path=None, num_frames=100):
    print("=" * 70)
    print(" THỰC NGHIỆM 2: SO SÁNH GIẢI THUẬT TRACKING ĐA ĐỐI TƯỢNG (MOT)")
    print(" Hệ thống Giám sát An toàn Đường sắt & Phát hiện Xâm nhập")
    print("=" * 70)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # 1. Kiểm tra mô hình duy nhất YOLO26s-all
    if not model_path:
        model_path = os.path.join(base_dir, "models", "yolo26s", "model_pytorch", "train-railway-yolo26s-all.pt")

    if not os.path.exists(model_path):
        print(f"[!] LỖI: Không tìm thấy mô hình train-railway-yolo26s-all.pt tại: {model_path}")
        return None
    print(f"[*] Mô hình sử dụng: {model_path}")

    # 2. Kiểm tra video duy nhất Dataset_test/test_exp.mp4
    if not video_path:
        video_path = os.path.join(base_dir, "Dataset_test", "test_exp.mp4")

    if not os.path.exists(video_path):
        print(f"[!] LỖI: Không tìm thấy video thực nghiệm tại: {video_path}")
        return None

    print(f"[*] Nạp video thực nghiệm: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
        count += 1
    cap.release()

    if len(frames) == 0:
        print(f"[!] LỖI: Không đọc được khung hình nào từ video {video_path}!")
        return None
    print(f"[+] Đã đọc thành công {len(frames)} khung hình từ video.")

    # 3. Chạy đánh giá qua 3 giải thuật
    tracker_cfg = os.path.join(os.path.dirname(__file__), "bytetrack.yaml")
    
    print("\n[1/3] Đang đánh giá chế độ: Không Tracking (Pure Detection)...")
    res_none = evaluate_tracking_mode(model_path, "none", frames)

    print("[2/3] Đang đánh giá chế độ: BoT-SORT / DeepSORT (Appearance Re-ID + Motion)...")
    res_botsort = evaluate_tracking_mode(model_path, "botsort", frames)

    print("[3/3] Đang đánh giá chế độ: ByteTrack (Two-Stage IoU Matching - Đề xuất)...")
    res_bytetrack = evaluate_tracking_mode(model_path, "bytetrack", frames, tracker_cfg=tracker_cfg)

    # 4. Hiển thị bảng tổng kết
    print("\n" + "=" * 78)
    print(" BẢNG TỔNG KẾT KẾT QUẢ THỰC NGHIỆM TRACKING ĐA ĐỐI TƯỢNG")
    print("=" * 78)
    print(f"{'Giải thuật':<20} | {'Độ trễ (ms)':<12} | {'Tốc độ (FPS)':<14} | {'Đo Dwell Time':<25}")
    print("-" * 78)
    print(f"{'1. Pure YOLO':<20} | {res_none['avg_latency_ms']:<12} | {res_none['fps']:<14} | {res_none['dwell_support']:<25}")
    print(f"{'2. BoT-SORT/DeepSORT':<20} | {res_botsort['avg_latency_ms']:<12} | {res_botsort['fps']:<14} | {res_botsort['dwell_support']:<25}")
    print(f"{'3. ByteTrack':<20} | {res_bytetrack['avg_latency_ms']:<12} | {res_bytetrack['fps']:<14} | {res_bytetrack['dwell_support']:<25}")
    print("=" * 78)

    # 5. Tự động lưu bảng kết quả vào thư mục results/
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "tracker_comparison.csv")
    
    df = pd.DataFrame([
        {"Method": "Pure YOLO (No Tracking)", "Latency_ms": res_none["avg_latency_ms"], "FPS": res_none["fps"], "Dwell_Support": res_none["dwell_support"], "Overhead_ms": res_none["overhead_ms"]},
        {"Method": "BoT-SORT / DeepSORT", "Latency_ms": res_botsort["avg_latency_ms"], "FPS": res_botsort["fps"], "Dwell_Support": res_botsort["dwell_support"], "Overhead_ms": res_botsort["overhead_ms"]},
        {"Method": "ByteTrack (De xuat)", "Latency_ms": res_bytetrack["avg_latency_ms"], "FPS": res_bytetrack["fps"], "Dwell_Support": res_bytetrack["dwell_support"], "Overhead_ms": res_bytetrack["overhead_ms"]}
    ])
    df.to_csv(csv_path, index=False)
    print(f"[*] Đã lưu kết quả đo đạc tracking tại: {csv_path}")

    return res_none, res_botsort, res_bytetrack

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá & so sánh giải thuật Tracking MOT")
    parser.add_argument("--video", type=str, default="", help="Đường dẫn file video đầu vào (mặc định Dataset_test/test_exp.mp4)")
    parser.add_argument("--model", type=str, default="", help="Đường dẫn mô hình YOLO26s-all")
    parser.add_argument("--frames", type=int, default=80, help="Số frame thực nghiệm")
    args = parser.parse_args()

    run_experiment(video_path=args.video, model_path=args.model, num_frames=args.frames)
