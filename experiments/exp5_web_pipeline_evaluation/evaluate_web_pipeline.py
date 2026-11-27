"""
Thực nghiệm 5: Đo lường hiệu năng thời gian chạy luồng xử lý Web
và phân tích định lượng các ca báo sót / lọc bỏ sự kiện (RailSafe-CV v1.3.0).

Đo trực tiếp toàn bộ chuỗi xử lý khép kín tương đương luồng phát video Web
(app/services/detector.py -> generate_frames):
    1. cap.read()           : Đọc khung hình đầu vào
    2. model.track()        : Suy luận YOLO + ByteTrack MOT (tự động chọn phần cứng)
    3. process_and_draw()   : Kiểm tra ROI đa giác, logic Dwell Time, vẽ lớp phủ
    4. cv2.imencode('.jpg') : Mã hóa khung hình MJPEG (chất lượng JPEG = 80)

Cách chạy từ thư mục gốc dự án:
    python experiments/exp5_web_pipeline_evaluation/evaluate_web_pipeline.py
    python experiments/exp5_web_pipeline_evaluation/evaluate_web_pipeline.py --video path/to/demo1.mp4
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "recordings" / "25-11-2026" / "25-11-2026_09h55m59s_den_09h56m25s.mp4"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.services.detector import (
    MIN_CONFIRM_SECONDS,
    MIN_EVENT_DURATION,
    load_yolo_model,
    translate_class,
)
from app.services.tracker import (
    check_roi_intersection,
    reset_tracker,
    update_dwell_and_evaluate_violation,
)
from app.utils.video_utils import draw_roi as util_draw_roi


def compute_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "max": round(float(arr.max()), 2),
    }


def bbox_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_web_pipeline(video_path: Path, out_dir: Path, warmup: int = 20, max_frames: int = 0):
    if not video_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy video đầu vào: {video_path}\n"
            "Hãy truyền đường dẫn video thực tế bằng tham số --video <path_to_video.mp4>."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tải mô hình bằng đúng bộ nạp thích ứng phần cứng của ứng dụng Web
    model, device, active_model_name = load_yolo_model()
    roi_polygon = config.load_roi_from_file()
    conf_web = config.load_conf_from_file()
    selected_classes = set(config.load_classes_from_file())
    tracker_cfg = config.BYTETRACK_FILE

    print(f"[INFO] Active model : {active_model_name} | Device: {device}")
    print(f"[INFO] Web config   : conf={conf_web}, classes={sorted(selected_classes)}, roi_pts={len(roi_polygon)}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video        : {video_path.name} | {w}x{h} @ {fps:.1f} FPS | {total_frames} frames")

    pts = np.array([[int(p["x"] * w), int(p["y"] * h)] for p in roi_polygon], np.int32)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [pts], 255)

    t_reads, t_infers, t_posts, t_jpegs, t_totals = [], [], [], [], []
    untracked_in_roi_count = 0
    untracked_examples = []
    track_history = {}
    roi_dwell_train_app = {}
    roi_dwell_no_train = {}
    track_roi_stats = {}

    reset_tracker(model)
    frame_idx = 0

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret or frame is None:
            break
        frame_idx += 1
        t_sec = frame_idx / fps

        # Bước 2: Suy luận YOLO + theo dõi ByteTrack (đồng nhất với detector.py)
        t_inf0 = time.perf_counter()
        results = model.track(
            source=frame,
            conf=conf_web,
            imgsz=640,
            device=device,
            verbose=False,
            persist=True,
            tracker=tracker_cfg,
        )
        t_inf1 = time.perf_counter()

        # Bước 3: Kiểm tra ROI, cập nhật Dwell Time và vẽ lớp phủ (đồng nhất với process_and_draw)
        t_post0 = time.perf_counter()
        vis = util_draw_roi(frame.copy(), roi_polygon)
        _rec_frame = vis.copy()  # Mô phỏng bước nhân bản khung hình cho bộ đệm Pre-roll

        if results and len(results) > 0 and results[0].boxes is not None:
            names = results[0].names
            for box in results[0].boxes:
                cls_id = int(box.cls)
                obj_name = names[cls_id]
                b_conf = float(box.conf[0])
                if obj_name not in selected_classes and obj_name != "train":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                intersect = check_roi_intersection((x1, y1, x2, y2), roi_mask, pts, (h, w))

                # Lọc bỏ hộp chưa được tracker gán ID (box.id is None)
                if box.id is None:
                    if intersect and obj_name != "train":
                        untracked_in_roi_count += 1
                        if len(untracked_examples) < 15:
                            untracked_examples.append(
                                {
                                    "frame": frame_idx,
                                    "t_sec": round(t_sec, 2),
                                    "cls": obj_name,
                                    "conf": round(b_conf, 3),
                                    "box": [x1, y1, x2, y2],
                                }
                            )
                    continue

                tid = str(int(box.id[0]))
                if obj_name == "train":
                    continue

                track_history.setdefault(tid, []).append(
                    (frame_idx, t_sec, obj_name, b_conf, [x1, y1, x2, y2], intersect)
                )

                if intersect:
                    is_viol, label_suffix = update_dwell_and_evaluate_violation(
                        tid, t_sec, "train_approaching", 5.0, roi_dwell_train_app, MIN_CONFIRM_SECONDS
                    )
                    update_dwell_and_evaluate_violation(
                        tid, t_sec, "no_train", 5.0, roi_dwell_no_train, MIN_CONFIRM_SECONDS
                    )
                    dwell_now = t_sec - roi_dwell_train_app[tid]["enter_time"]
                    st = track_roi_stats.setdefault(
                        tid,
                        {
                            "cls": obj_name,
                            "first_f": frame_idx,
                            "last_f": frame_idx,
                            "first_t": t_sec,
                            "last_t": t_sec,
                            "frames_in_roi": 0,
                            "max_dwell": 0.0,
                        },
                    )
                    st["last_f"] = frame_idx
                    st["last_t"] = t_sec
                    st["frames_in_roi"] += 1
                    st["max_dwell"] = max(st["max_dwell"], dwell_now)

                    color = (0, 0, 255) if is_viol else (0, 215, 255)
                    label = f"{translate_class(obj_name)} - ID: {tid} {label_suffix}"
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        vis,
                        label,
                        (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    if tid in roi_dwell_train_app and t_sec - roi_dwell_train_app[tid]["last_seen"] > 1.5:
                        del roi_dwell_train_app[tid]
                    if tid in roi_dwell_no_train and t_sec - roi_dwell_no_train[tid]["last_seen"] > 1.5:
                        del roi_dwell_no_train[tid]

        cv2.putText(vis, "25-11-2026 09:56:00", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        t_post1 = time.perf_counter()

        # Bước 4: Mã hóa JPEG cho luồng MJPEG (chất lượng 80 như detector.py)
        t_jpg0 = time.perf_counter()
        cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        t_jpg1 = time.perf_counter()

        if frame_idx > warmup:
            t_reads.append((t1 - t0) * 1000.0)
            t_infers.append((t_inf1 - t_inf0) * 1000.0)
            t_posts.append((t_post1 - t_post0) * 1000.0)
            t_jpegs.append((t_jpg1 - t_jpg0) * 1000.0)
            t_totals.append((t_jpg1 - t0) * 1000.0)

        if max_frames > 0 and frame_idx >= max_frames:
            break

    cap.release()

    s_read = compute_stats(t_reads)
    s_inf = compute_stats(t_infers)
    s_post = compute_stats(t_posts)
    s_jpg = compute_stats(t_jpegs)
    s_tot = compute_stats(t_totals)
    eff_fps = round(1000.0 / s_tot["mean"], 1) if s_tot["mean"] > 0 else 0.0

    # Phân tích 3 nhóm nguyên nhân báo sót / lọc bỏ trên ROI
    # A. Detector mất box (khoảng trống liên tiếp của cùng Track ID trong ROI)
    lost_box_cases = []
    for tid, hist in track_history.items():
        roi_frames = [entry for entry in hist if entry[5]]
        for i in range(1, len(roi_frames)):
            gap = roi_frames[i][0] - roi_frames[i - 1][0] - 1
            if 1 <= gap <= 25:
                lost_box_cases.append(
                    {
                        "tid": tid,
                        "cls": roi_frames[i][2],
                        "before_f": roi_frames[i - 1][0],
                        "after_f": roi_frames[i][0],
                        "gap_frames": gap,
                        "t_sec": round(roi_frames[i - 1][1], 2),
                        "box_before": roi_frames[i - 1][4],
                        "box_after": roi_frames[i][4],
                    }
                )
    lost_box_cases.sort(key=lambda item: -item["gap_frames"])

    # B. Tracker đổi ID (track A kết thúc trong ROI, track B cùng lớp xuất hiện gần đó sau 1..12 khung)
    ends, starts = [], []
    for tid, hist in track_history.items():
        roi_h = [entry for entry in hist if entry[5]]
        if roi_h:
            starts.append((tid, roi_h[0]))
            ends.append((tid, roi_h[-1]))

    id_switches = []
    for tid1, e1 in ends:
        for tid2, s2 in starts:
            if tid1 == tid2 or e1[2] != s2[2]:
                continue
            df = s2[0] - e1[0]
            if 1 <= df <= 12:
                ov = bbox_iou(e1[4], s2[4])
                c1 = ((e1[4][0] + e1[4][2]) / 2.0, (e1[4][1] + e1[4][3]) / 2.0)
                c2 = ((s2[4][0] + s2[4][2]) / 2.0, (s2[4][1] + s2[4][3]) / 2.0)
                dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                if ov >= 0.25 or dist < 70.0:
                    id_switches.append(
                        {
                            "old_id": tid1,
                            "new_id": tid2,
                            "cls": e1[2],
                            "end_f": e1[0],
                            "start_f": s2[0],
                            "gap_f": df,
                            "t_sec": round(e1[1], 2),
                            "iou": round(ov, 2),
                            "dist_px": round(dist, 1),
                            "old_box": e1[4],
                            "new_box": s2[4],
                        }
                    )

    # C. Phân loại thời gian tích lũy trong ROI (<0.5s, 0.5..1.0s, >=1.0s)
    short_under_05, short_05_to_10, valid_over_10 = [], [], []
    for tid, st in track_roi_stats.items():
        dur = round(st["last_t"] - st["first_t"], 2)
        entry = {
            "tid": tid,
            "cls": st["cls"],
            "first_f": st["first_f"],
            "last_f": st["last_f"],
            "t_start": round(st["first_t"], 2),
            "t_end": round(st["last_t"], 2),
            "dur_s": dur,
            "frames_in_roi": st["frames_in_roi"],
        }
        if dur < MIN_CONFIRM_SECONDS:
            short_under_05.append(entry)
        elif dur < MIN_EVENT_DURATION:
            short_05_to_10.append(entry)
        else:
            valid_over_10.append(entry)

    # Xuất bảng CSV hiệu năng thời gian chạy
    csv_path = out_dir / "web_pipeline_runtime.csv"
    tot_mean = s_tot["mean"] or 1.0
    rows = [
        ("1. Doc khung hinh (cap.read)", s_read["mean"], s_read["median"], s_read["p95"], round(s_read["mean"] / tot_mean * 100, 1)),
        ("2. Suy luan + Tracking (model.track)", s_inf["mean"], s_inf["median"], s_inf["p95"], round(s_inf["mean"] / tot_mean * 100, 1)),
        ("3. Nghiep vu ROI + Canh bao + Ve phu", s_post["mean"], s_post["median"], s_post["p95"], round(s_post["mean"] / tot_mean * 100, 1)),
        ("4. Ma hoa MJPEG (JPEG quality=80)", s_jpg["mean"], s_jpg["median"], s_jpg["p95"], round(s_jpg["mean"] / tot_mean * 100, 1)),
        ("Tong toan luong Web", s_tot["mean"], s_tot["median"], s_tot["p95"], 100.0),
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "mean_ms", "median_ms", "p95_ms", "share_pct"])
        writer.writerows(rows)

    # Xuất tệp JSON chi tiết phục vụ kiểm tra lại
    json_path = out_dir / "web_pipeline_diagnostics.json"
    summary = {
        "model": active_model_name,
        "device": device,
        "conf": conf_web,
        "video": video_path.name,
        "fps_video": fps,
        "resolution": [w, h],
        "measured_frames": len(t_totals),
        "warmup_frames": warmup,
        "timing": {
            "read": s_read,
            "infer_track": s_inf,
            "post_draw": s_post,
            "jpeg": s_jpg,
            "total": s_tot,
            "processing_fps": eff_fps,
        },
        "untracked_in_roi_count": untracked_in_roi_count,
        "untracked_examples": untracked_examples[:10],
        "lost_box_cases": lost_box_cases[:20],
        "id_switches": id_switches[:15],
        "short_under_05": short_under_05,
        "short_05_to_10": short_05_to_10,
        "valid_over_10": valid_over_10,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== KET QUA DO LUONG WEB PIPELINE ===")
    for name, m, med, p95, pct in rows:
        print(f"  {name:<38} | mean={m:>6.2f} ms | median={med:>6.2f} ms | P95={p95:>6.2f} ms | {pct:>5.1f}%")
    print(f"  Toc do xu ly thuc te (Processing FPS)  : {eff_fps} FPS")
    print(f"  Untracked (box.id is None) trong ROI   : {untracked_in_roi_count}")
    print(f"  Doan mat box trong ROI (1..25 frames)  : {len(lost_box_cases)}")
    print(f"  Ca doi ID trong ROI (ID switches)      : {len(id_switches)}")
    print(f"  Quy dao < 0.5s / 0.5-1.0s / >= 1.0s    : {len(short_under_05)} / {len(short_05_to_10)} / {len(valid_over_10)}")
    print(f"[INFO] Da luu CSV  : {csv_path}")
    print(f"[INFO] Da luu JSON : {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Đo hiệu năng thời gian chạy luồng Web và phân tích ca báo sót.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Đường dẫn video kiểm thử.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Thư mục lưu CSV/JSON kết quả.")
    parser.add_argument("--warmup", type=int, default=20, help="Số khung hình khởi động bỏ qua khi đo FPS.")
    parser.add_argument("--max-frames", type=int, default=0, help="Giới hạn số khung hình (0 = chạy toàn bộ video).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_web_pipeline(args.video, args.output_dir, warmup=args.warmup, max_frames=args.max_frames)
