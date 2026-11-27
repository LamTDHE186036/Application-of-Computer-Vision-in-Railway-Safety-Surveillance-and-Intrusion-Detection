"""
Công cụ đánh giá ngoại tuyến ở cấp độ sự kiện (Proposed Event-Level Evaluation Protocol)
tương ứng với quy trình đề xuất tại Mục 4.3.4 của báo cáo RailSafe-CV.

Tính toán các chỉ số chuẩn khi có tệp nhãn sự kiện thực tế (ground truth) và
tệp sự kiện dự đoán từ hệ thống (predicted events):
    - Temporal IoU (tIoU) theo ngưỡng theta_t (mặc định 0.3)
    - Khớp 1-1 tham lam theo tIoU giảm dần -> TP, FP, FN
    - Event Precision, Event Recall, Event F1-Score
    - False Alarm Rate trên mỗi giờ giám sát (FAR = FP / T_hour)
    - Alert Detection Delay trung bình (Delta t_alert = t_hat_start - t_start)

Định dạng JSON mỗi sự kiện:
    [
      {"t_start": 10.2, "t_end": 18.5, "cls": "person", "mode": "no_train"}
    ]

Cách chạy kiểm tra nhanh (self-test với dữ liệu mẫu minh họa):
    python experiments/exp5_web_pipeline_evaluation/evaluate_events_offline.py --self-test

Cách chạy chấm điểm trên tập video đã gán nhãn:
    python experiments/exp5_web_pipeline_evaluation/evaluate_events_offline.py \
        --gt gt_events.json --pred pred_events.json --duration-hours 1.0 --tiou-threshold 0.3
"""

import argparse
import json
from pathlib import Path


def temporal_iou(e1: dict, e2: dict) -> float:
    s1, t1 = float(e1["t_start"]), float(e1["t_end"])
    s2, t2 = float(e2["t_start"]), float(e2["t_end"])
    inter = max(0.0, min(t1, t2) - max(s1, s2))
    union = max(t1, t2) - min(s1, s2)
    return inter / union if union > 0.0 else 0.0


def evaluate_events(
    gt_events: list,
    pred_events: list,
    tiou_threshold: float = 0.3,
    duration_hours: float = 1.0,
    match_class: bool = True,
) -> dict:
    """Khớp sự kiện 1-1 theo tIoU và tính Event Precision/Recall/F1, FAR, Alert Delay."""
    candidates = []
    for gi, gt in enumerate(gt_events):
        for pi, pr in enumerate(pred_events):
            if match_class and gt.get("cls") and pr.get("cls") and gt["cls"] != pr["cls"]:
                continue
            score = temporal_iou(gt, pr)
            if score >= tiou_threshold:
                candidates.append((score, gi, pi))

    # Khớp 1-1 ưu tiên các cặp có tIoU cao nhất
    candidates.sort(key=lambda item: -item[0])
    matched_gt, matched_pred = set(), set()
    matched_pairs = []
    for score, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        delay = float(pred_events[pi]["t_start"]) - float(gt_events[gi]["t_start"])
        matched_pairs.append({"gt_idx": gi, "pred_idx": pi, "tiou": round(score, 4), "delay_s": round(delay, 3)})

    tp = len(matched_pairs)
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far = fp / duration_hours if duration_hours > 0 else 0.0
    mean_delay = sum(p["delay_s"] for p in matched_pairs) / tp if tp > 0 else None

    return {
        "tiou_threshold": tiou_threshold,
        "duration_hours": duration_hours,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "event_precision": round(precision, 4),
        "event_recall": round(recall, 4),
        "event_f1": round(f1, 4),
        "far_per_hour": round(far, 4),
        "mean_alert_delay_s": round(mean_delay, 3) if mean_delay is not None else None,
        "matches": matched_pairs,
    }


def run_self_test():
    """Kiểm chứng logic hàm chấm điểm trên 3 sự kiện giả lập."""
    gt_sample = [
        {"t_start": 10.0, "t_end": 18.0, "cls": "person", "mode": "no_train"},
        {"t_start": 30.0, "t_end": 35.0, "cls": "motorcycle", "mode": "train_approaching"},
    ]
    pred_sample = [
        {"t_start": 10.6, "t_end": 18.2, "cls": "person", "mode": "no_train"},
        {"t_start": 50.0, "t_end": 52.0, "cls": "car", "mode": "train_approaching"},
    ]
    result = evaluate_events(gt_sample, pred_sample, tiou_threshold=0.3, duration_hours=0.5)
    assert result["TP"] == 1 and result["FP"] == 1 and result["FN"] == 1
    assert abs(result["event_precision"] - 0.5) < 1e-6
    assert abs(result["event_recall"] - 0.5) < 1e-6
    assert abs(result["far_per_hour"] - 2.0) < 1e-6
    assert abs(result["mean_alert_delay_s"] - 0.6) < 1e-6
    print("[SELF-TEST PASSED] Ket qua kiem thu ham tinh tIoU / Precision / Recall / F1 / FAR / Delay:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá ngoại tuyến cấp độ sự kiện (tIoU, Event P/R/F1, FAR, Delay).")
    parser.add_argument("--gt", type=Path, help="Đường dẫn tệp JSON nhãn sự kiện chuẩn (ground truth).")
    parser.add_argument("--pred", type=Path, help="Đường dẫn tệp JSON sự kiện cảnh báo của hệ thống.")
    parser.add_argument("--tiou-threshold", type=float, default=0.3, help="Ngưỡng tIoU khớp sự kiện (mặc định 0.3).")
    parser.add_argument("--duration-hours", type=float, default=1.0, help="Tổng thời lượng giám sát (giờ) để tính FAR.")
    parser.add_argument("--ignore-class", action="store_true", help="Bỏ qua so khớp nhãn lớp khi tính tIoU.")
    parser.add_argument("--self-test", action="store_true", help="Chạy kiểm thử nhanh với dữ liệu mẫu.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        run_self_test()
    elif args.gt and args.pred:
        gt_data = json.loads(args.gt.read_text(encoding="utf-8"))
        pred_data = json.loads(args.pred.read_text(encoding="utf-8"))
        report = evaluate_events(
            gt_data,
            pred_data,
            tiou_threshold=args.tiou_threshold,
            duration_hours=args.duration_hours,
            match_class=not args.ignore_class,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Hãy truyền --self-test để kiểm tra nhanh, hoặc --gt <gt.json> --pred <pred.json> để chấm điểm.")
