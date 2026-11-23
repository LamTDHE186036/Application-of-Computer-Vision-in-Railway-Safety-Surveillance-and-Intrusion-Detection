"""
Thực nghiệm 1: Đánh giá & So sánh Kiến trúc Mô hình Nhận diện (YOLO11s vs YOLO26s)
So sánh 2 mô hình huấn luyện trên dữ liệu an toàn đường sắt:
1. YOLO11s (Baseline - train-railway-yolo11s-all.pt)
2. YOLO26s (Proposed / Selected - train-railway-yolo26s-all.pt)

Đánh giá chéo trên các tập dữ liệu kiểm thử độc lập tại Dataset_test:
- Dataset_test/roboflow_crossing/data.yaml (Đường ngang giao cắt)
- Dataset_test/rcd_ground/data.yaml        (Đường ray mặt đất)
- Dataset_test/rcd_elevated/data.yaml      (Đường ray trên cao)
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Thiết lập đường dẫn cơ sở
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP1_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(EXP1_DIR, "results")

def resolve_model_path(model_filename, research_subfolder, models_subfolder):
    """Tự động dò tìm đường dẫn trọng số hợp lệ giữa thư mục research và models"""
    candidates = [
        os.path.join(BASE_DIR, "models", models_subfolder, "model_pytorch", f"train-railway-{models_subfolder}-all.pt"),
        os.path.join(BASE_DIR, "models", models_subfolder, "model_pytorch", model_filename),
        os.path.join(BASE_DIR, "research", "finetune", f"train_{models_subfolder}_all", f"train-railway-{models_subfolder}-all.pt"),
        os.path.join(BASE_DIR, "research", "finetune", f"train_{models_subfolder}_all", "weights", "best.pt"),
        os.path.join(BASE_DIR, "research", "finetune", research_subfolder, model_filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]

MODELS = {
    "YOLO11s": resolve_model_path("yolo11s.pt", "train_yolo11s_all", "yolo11s"),
    "YOLO26s": resolve_model_path("yolo26s.pt", "train_yolo26s_all", "yolo26s")
}

DATASETS = {
    "roboflow_crossing": os.path.join(BASE_DIR, "Dataset_test", "roboflow_crossing", "data.yaml"),
    "rcd_ground": os.path.join(BASE_DIR, "Dataset_test", "rcd_ground", "data.yaml"),
    "rcd_elevated": os.path.join(BASE_DIR, "Dataset_test", "rcd_elevated", "data.yaml")
}

def extract_per_class_metrics(val_results, c_idx):
    """Trích xuất chính xác cả mAP50 và mAP50-95 cho từng lớp đối tượng"""
    map50_95 = 0.0
    map50 = 0.0

    # val_results.box.maps chứa mAP50-95 của từng class
    if hasattr(val_results.box, "maps") and c_idx < len(val_results.box.maps):
        map50_95 = float(val_results.box.maps[c_idx])

    # val_results.box.all_ap có kích thước (num_classes, 10) tương ứng IoU 0.50 -> 0.95
    if hasattr(val_results.box, "all_ap") and val_results.box.all_ap is not None:
        all_ap = np.asarray(val_results.box.all_ap)
        if all_ap.ndim == 2 and c_idx < all_ap.shape[0]:
            map50 = float(all_ap[c_idx, 0])
        else:
            map50 = map50_95
    else:
        map50 = map50_95

    return map50, map50_95

def run_evaluation(batch_size=1, imgsz=640, device="cpu"):
    """
    Thực thi validation trực tiếp trên Dataset_test cho YOLO11s_all và YOLO26s_all.
    Mặc định batch_size=1 để đo chính xác độ trễ suy luận (Inference_ms) thời gian thực.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows = []
    class_rows = []

    print("=" * 85)
    print(" THỰC NGHIỆM 1: ĐÁNH GIÁ ĐỐI ĐẦU YOLO11s VS YOLO26s TRÊN DATASET_TEST")
    print(f"[*] Thiết bị suy luận : {device}")
    print(f"[*] Kích thước ảnh    : {imgsz}x{imgsz}")
    print(f"[*] Kích thước batch  : {batch_size}")
    print("=" * 85)

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"[!] Cảnh báo: Không tìm thấy file trọng số cho {model_name} tại {model_path}")
            continue

        print(f"\n>>> Đang nạp mô hình {model_name}: {model_path}")
        model = YOLO(model_path)

        for ds_name, ds_yaml in DATASETS.items():
            if not os.path.exists(ds_yaml):
                print(f"[!] Bỏ qua tập dữ liệu {ds_name} (Không tìm thấy {ds_yaml})")
                continue

            print(f"[*] Đang đánh giá {model_name} trên tập '{ds_name}'...")
            t0 = time.perf_counter()

            val_results = model.val(
                data=ds_yaml,
                batch=batch_size,
                imgsz=imgsz,
                device=device,
                plots=True,
                save=True,
                project=RESULTS_DIR,
                name=f"{model_name}_{ds_name}",
                exist_ok=True,
                verbose=False
            )
            val_time = time.perf_counter() - t0

            # Lấy các chỉ số tổng hợp
            p = float(val_results.box.mp)
            r = float(val_results.box.mr)
            map50 = float(val_results.box.map50)
            map50_95 = float(val_results.box.map)
            speed_inference = float(val_results.speed.get("inference", 0.0))
            speed_preprocess = float(val_results.speed.get("preprocess", 0.0))
            speed_postprocess = float(val_results.speed.get("postprocess", 0.0))
            total_latency = speed_preprocess + speed_inference + speed_postprocess

            # Lấy tổng số lượng nhãn đối tượng (instances) để tính trung bình có trọng số nếu cần
            num_targets = int(val_results.nt_per_class.sum()) if getattr(val_results, "nt_per_class", None) is not None else 1

            summary_rows.append({
                "Model": model_name,
                "Dataset": ds_name,
                "Targets": num_targets,
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "mAP50": round(map50, 4),
                "mAP50-95": round(map50_95, 4),
                "Inference_ms": round(speed_inference, 2),
                "Total_Latency_ms": round(total_latency, 2),
                "Validation_Time_s": round(val_time, 2)
            })

            # Lấy chỉ số chi tiết theo từng lớp (Per-class)
            dataset_classes = val_results.names
            for c_idx, c_name in dataset_classes.items():
                c_map50, c_map50_95 = extract_per_class_metrics(val_results, c_idx)
                class_rows.append({
                    "Model": model_name,
                    "Dataset": ds_name,
                    "Class": c_name,
                    "mAP50": round(c_map50, 4),
                    "mAP50-95": round(c_map50_95, 4)
                })

            print(
                f"    [+] Xong {model_name} - {ds_name}: "
                f"P={p*100:.2f}%, R={r*100:.2f}%, mAP50={map50*100:.2f}%, "
                f"mAP50-95={map50_95*100:.2f}%, Inference={speed_inference:.2f}ms"
            )

    if not summary_rows:
        print("[!] Không có dữ liệu kết quả nào được ghi nhận. Vui lòng kiểm tra lại đường dẫn!")
        return None, None

    df_summary = pd.DataFrame(summary_rows)
    df_class = pd.DataFrame(class_rows)

    # Lưu kết quả chi tiết
    summary_csv = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    class_csv = os.path.join(RESULTS_DIR, "comparison_per_class.csv")
    df_summary.to_csv(summary_csv, index=False)
    df_class.to_csv(class_csv, index=False)

    # Tính toán bảng chỉ số trung bình (Macro Average)
    metric_cols = ["Precision", "Recall", "mAP50", "mAP50-95", "Inference_ms", "Total_Latency_ms", "Validation_Time_s"]
    df_avg = df_summary.groupby("Model", as_index=False)[metric_cols].mean().round(4)
    avg_csv = os.path.join(RESULTS_DIR, "comparison_average.csv")
    df_avg.to_csv(avg_csv, index=False)

    print("\n" + "=" * 95)
    print(" BẢNG TỔNG HỢP SO SÁNH TRUNG BÌNH 3 TẬP DỮ LIỆU KIỂM THỬ (MACRO AVERAGE METRICS)")
    print("=" * 95)
    print(f"{'Mô hình':<12} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<14} | {'Suy luận (ms)':<15}")
    print("-" * 95)
    for _, row in df_avg.iterrows():
        p_pct = f"{row['Precision']*100:.2f}%"
        r_pct = f"{row['Recall']*100:.2f}%"
        map50_pct = f"{row['mAP50']*100:.2f}%"
        map95_pct = f"{row['mAP50-95']*100:.2f}%"
        lat = f"{row['Inference_ms']:.2f} ms/ảnh"
        print(f"{row['Model']:<12} | {p_pct:<10} | {r_pct:<10} | {map50_pct:<10} | {map95_pct:<14} | {lat:<15}")
    print("=" * 95)
    print(f"[*] Đã lưu bảng tổng kết chi tiết tại : {summary_csv}")
    print(f"[*] Đã lưu bảng chi tiết từng lớp tại : {class_csv}")
    print(f"[*] Đã lưu bảng trung bình tại        : {avg_csv}")

    return df_summary, df_avg

def display_saved_results():
    """Hiển thị nhanh kết quả đã đo đạc nếu file CSV đã tồn tại"""
    avg_csv = os.path.join(RESULTS_DIR, "comparison_average.csv")
    summary_csv = os.path.join(RESULTS_DIR, "comparison_summary.csv")

    if not os.path.exists(avg_csv):
        print("[*] Chưa có kết quả đo sẵn, bắt đầu chạy đánh giá trên Dataset_test...")
        return run_evaluation()

    df_avg = pd.read_csv(avg_csv)
    print("=" * 95)
    print(" KẾT QUẢ SO SÁNH TRUNG BÌNH ĐÃ LƯU (YOLO11s VS YOLO26s)")
    print("=" * 95)
    print(df_avg.to_string(index=False))
    print("=" * 95)
    return pd.read_csv(summary_csv), df_avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh YOLO11s vs YOLO26s trên Dataset_test")
    parser.add_argument("--eval", action="store_true", help="Chạy lại toàn bộ quá trình validation trên Dataset_test")
    parser.add_argument("--batch", type=int, default=1, help="Kích thước batch (Mặc định 1 để đo chuẩn latency)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị thực thi (cpu, 0, intel:gpu)")
    args = parser.parse_args()

    if args.eval or not os.path.exists(os.path.join(RESULTS_DIR, "comparison_average.csv")):
        run_evaluation(batch_size=args.batch, imgsz=args.imgsz, device=args.device)
    else:
        display_saved_results()