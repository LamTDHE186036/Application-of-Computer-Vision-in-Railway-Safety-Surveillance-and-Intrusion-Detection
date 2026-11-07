"""
Thực nghiệm 1: Đánh giá & So sánh Kiến trúc Mô hình Nhận diện (YOLO11s-all vs YOLO26s-all)
So sánh 2 mô hình đã finetune toàn diện (5 classes) trên tập dữ liệu an toàn đường sắt:
1. YOLO11s_all (Trọng số: research/finetune/train_yolo11s_all/train-railway-yolo11s-all.pt)
2. YOLO26s_all (Trọng số: research/finetune/train_yolo26s_all/train-railway-yolo26s-all.pt)

Đánh giá chéo trên 3 tập dữ liệu kiểm thử độc lập tại Dataset_test:
- Dataset_test/roboflow_crossing/data.yaml (Đường ngang giao cắt)
- Dataset_test/rcd_ground/data.yaml        (Đường ray mặt đất)
- Dataset_test/rcd_elevated/data.yaml      (Đường ray trên cao)
"""

import os
import time
import argparse
import pandas as pd
from ultralytics import YOLO

# Thiết lập đường dẫn cơ sở
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP1_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(EXP1_DIR, "results")

MODELS = {
    "YOLO11s_all": os.path.join(BASE_DIR, "research", "finetune", "train_yolo11s_all", "train-railway-yolo11s-all.pt"),
    "YOLO26s_all": os.path.join(BASE_DIR, "research", "finetune", "train_yolo26s_all", "train-railway-yolo26s-all.pt")
}

DATASETS = {
    "roboflow_crossing": os.path.join(BASE_DIR, "Dataset_test", "roboflow_crossing", "data.yaml"),
    "rcd_ground": os.path.join(BASE_DIR, "Dataset_test", "rcd_ground", "data.yaml"),
    "rcd_elevated": os.path.join(BASE_DIR, "Dataset_test", "rcd_elevated", "data.yaml")
}

def run_evaluation(batch_size=8, imgsz=640, device="cpu"):
    """
    Thực thi validation trực tiếp trên Dataset_test cho YOLO11s_all và YOLO26s_all
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows = []
    class_rows = []

    print("=" * 80)
    print(" THỰC NGHIỆM 1: ĐÁNH GIÁ ĐỐI ĐẦU YOLO11s-ALL VS YOLO26s-ALL TRÊN DATASET_TEST")
    print(f"[*] Thiết bị suy luận : {device}")
    print(f"[*] Kích thước ảnh    : {imgsz}x{imgsz}")
    print(f"[*] Kích thước batch  : {batch_size}")
    print("=" * 80)

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
            t0 = time.time()
            
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
            val_time = time.time() - t0

            # Lấy các chỉ số tổng hợp
            p = float(val_results.box.mp)
            r = float(val_results.box.mr)
            map50 = float(val_results.box.map50)
            map50_95 = float(val_results.box.map)
            speed_inference = float(val_results.speed.get("inference", 0.0))

            summary_rows.append({
                "Model": model_name,
                "Dataset": ds_name,
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "mAP50": round(map50, 4),
                "mAP50-95": round(map50_95, 4),
                "Inference_ms": round(speed_inference, 2),
                "Validation_Time_s": round(val_time, 2)
            })

            # Lấy chỉ số chi tiết theo từng lớp (Per-class)
            maps_per_class = val_results.box.maps
            dataset_classes = val_results.names
            for c_idx, c_name in dataset_classes.items():
                c_map50 = float(maps_per_class[c_idx]) if c_idx < len(maps_per_class) else 0.0
                class_rows.append({
                    "Model": model_name,
                    "Dataset": ds_name,
                    "Class": c_name,
                    "mAP50": round(c_map50, 4)
                })

            print(f"    [+] Xong {model_name} - {ds_name}: P={p*100:.2f}%, R={r*100:.2f}%, mAP50={map50*100:.2f}%, Time={val_time:.1f}s")

    df_summary = pd.DataFrame(summary_rows)
    df_class = pd.DataFrame(class_rows)

    # Lưu kết quả chi tiết
    summary_csv = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    class_csv = os.path.join(RESULTS_DIR, "comparison_per_class.csv")
    df_summary.to_csv(summary_csv, index=False)
    df_class.to_csv(class_csv, index=False)

    # Tính toán bảng chỉ số trung bình (Average)
    df_avg = df_summary.groupby("Model")[["Precision", "Recall", "mAP50", "mAP50-95", "Inference_ms", "Validation_Time_s"]].mean().reset_index()
    avg_csv = os.path.join(RESULTS_DIR, "comparison_average.csv")
    df_avg.to_csv(avg_csv, index=False)

    print("\n" + "=" * 90)
    print(" BẢNG TỔNG HỢP SO SÁNH TRUNG BÌNH 3 TẬP DỮ LIỆU KIỂM THỬ (AVERAGE METRICS)")
    print("=" * 90)
    print(f"{'Mô hình':<10} | {'Precision':<12} | {'Recall':<10} | {'mAP@0.5':<12} | {'mAP@0.5:0.95':<14} | {'Độ trễ suy luận':<16}")
    print("-" * 90)
    for _, row in df_avg.iterrows():
        p_pct = f"{row['Precision']*100:.2f}%"
        r_pct = f"{row['Recall']*100:.2f}%"
        map50_pct = f"{row['mAP50']*100:.2f}%"
        map95_pct = f"{row['mAP50-95']*100:.2f}%"
        lat = f"{row['Inference_ms']:.1f} ms/ảnh"
        print(f"{row['Model']:<10} | {p_pct:<12} | {r_pct:<10} | {map50_pct:<12} | {map95_pct:<14} | {lat:<16}")
    print("=" * 90)
    print(f"[*] Đã lưu bảng tổng kết chi tiết tại: {summary_csv}")
    print(f"[*] Đã lưu bảng trung bình tại       : {avg_csv}")

    return df_summary, df_avg

def display_saved_results():
    """
    Hiển thị nhanh kết quả đã đo đạc nếu file CSV đã tồn tại
    """
    avg_csv = os.path.join(RESULTS_DIR, "comparison_average.csv")
    summary_csv = os.path.join(RESULTS_DIR, "comparison_summary.csv")

    if not os.path.exists(avg_csv):
        print("[*] Chưa có kết quả đo sẵn, bắt đầu chạy đánh giá trên Dataset_test...")
        return run_evaluation()

    df_avg = pd.read_csv(avg_csv)
    print("=" * 90)
    print(" KẾT QUẢ SO SÁNH TRUNG BÌNH ĐÃ LƯU (YOLO11s-ALL VS YOLO26s-ALL)")
    print("=" * 90)
    print(df_avg.to_string(index=False))
    print("=" * 90)
    return pd.read_csv(summary_csv), df_avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh YOLO11s-all vs YOLO26s-all trên Dataset_test")
    parser.add_argument("--eval", action="store_true", help="Chạy lại toàn bộ quá trình validation trên Dataset_test")
    parser.add_argument("--batch", type=int, default=8, help="Kích thước batch")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị thực thi (cpu, 0, intel:gpu)")
    args = parser.parse_args()

    if args.eval or not os.path.exists(os.path.join(RESULTS_DIR, "comparison_average.csv")):
        run_evaluation(batch_size=args.batch, device=args.device)
    else:
        display_saved_results()
