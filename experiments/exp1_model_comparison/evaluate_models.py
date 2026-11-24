"""
Thực nghiệm 1: Đánh giá & So sánh tất cả các Mô hình trong thư mục research/ trên dataset_test:
1. YOLO11s          (research/finetune/train_yolo11s/yolo11s.pt)
2. YOLO26s          (research/finetune/train_yolo26s/yolo26s.pt)
3. YOLO26s_freeze3  (research/finetune/train_yolo26s_freeze3/yolo26s_freeze3.pt)
4. YOLO26s_freeze5  (research/finetune/train_yolo26s_freeze5/yolo26s_freeze5.pt)
5. YOLO26s_freeze10 (research/finetune/train_yolo26s_freeze10/yolo26s_freeze10.pt)

Đánh giá chéo trên 3 tập dữ liệu kiểm thử độc lập tại dataset_test:
- dataset_test/roboflow_crossing/data.yaml (5 lớp: car, motorcycle, person, train, truck-bus)
- dataset_test/rcd_ground/data.yaml        (3 lớp: car, person, truck-bus)
- dataset_test/rcd_elevated/data.yaml      (4 lớp: car, person, train, truck-bus)

Cơ chế tự động đồng bộ khi đánh giá (MappedDetectionValidator):
- Tự động xử lý đường dẫn Roboflow (`../valid/images` -> `images`) bằng file YAML tạm mà không sửa `dataset_test`.
- Ánh xạ chính xác ID lớp 1-1 theo tên lớp giữa từng mô hình và từng tập dữ liệu
  (xử lý thứ tự ID đảo vị trí giữa `train`/`truck-bus` và lọc bỏ các lớp không có trong tập con).
"""

import os
import time
import tempfile
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torchvision
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix

# Thiết lập đường dẫn cơ sở
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP1_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(EXP1_DIR, "results")
PRETRAINED_DIR = os.path.join(EXP1_DIR, "weights")
FINETUNE_DIR = os.path.join(BASE_DIR, "research", "finetune")

# Ánh xạ riêng cho mô hình Pretrained COCO (80 lớp, trong đó tách riêng 'truck' và 'bus')
COCO_CLASS_MAP = {
    "truck": "truck-bus",
    "bus": "truck-bus",
}


def resolve_dataset_dir():
    """Tự động dò tìm thư mục dataset_test (không phân biệt chữ hoa/thường)"""
    for folder_name in ("dataset_test", "Dataset_test"):
        candidate = os.path.join(BASE_DIR, folder_name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(BASE_DIR, "dataset_test")


def first_existing_path(candidates):
    """Trả về đường dẫn đầu tiên tồn tại trên đĩa; nếu không có, trả về ứng viên đầu tiên"""
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def resolve_models():
    """
    Tự động dò tìm YOLO26s_Pretrained và tất cả các mô hình trong research/finetune/ theo thứ tự chuẩn:
    YOLO26s_Pretrained -> YOLO11s -> YOLO26s -> YOLO26s_freeze3 -> YOLO26s_freeze5 -> YOLO26s_freeze10
    (đồng thời tự động phát hiện thêm thư mục train_* mới nếu có).
    """
    known_models = [
        (
            "YOLO26s_Pretrained",
            "__pretrained__",
            [
                os.path.join(PRETRAINED_DIR, "yolo26s.pt"),
                os.path.join(BASE_DIR, "yolo26s.pt"),
                "yolo26s.pt",
            ],
        ),
        (
            "YOLO11s",
            "train_yolo11s",
            [
                os.path.join(FINETUNE_DIR, "train_yolo11s", "yolo11s.pt"),
                os.path.join(FINETUNE_DIR, "train_yolo11s", "train-railway-yolo11s", "weights", "best.pt"),
                os.path.join(BASE_DIR, "models", "yolo11s", "model_pytorch", "train-railway-yolo11s-all.pt"),
            ],
        ),
        (
            "YOLO26s",
            "train_yolo26s",
            [
                os.path.join(FINETUNE_DIR, "train_yolo26s", "yolo26s.pt"),
                os.path.join(FINETUNE_DIR, "train_yolo26s", "train-railway-yolo26s", "weights", "best.pt"),
                os.path.join(BASE_DIR, "models", "yolo26s", "model_pytorch", "train-railway-yolo26s-all.pt"),
            ],
        ),
        (
            "YOLO26s_freeze3",
            "train_yolo26s_freeze3",
            [
                os.path.join(FINETUNE_DIR, "train_yolo26s_freeze3", "yolo26s_freeze3.pt"),
                os.path.join(FINETUNE_DIR, "train_yolo26s_freeze3", "railway_freeze3_output", "weights", "best.pt"),
            ],
        ),
        (
            "YOLO26s_freeze5",
            "train_yolo26s_freeze5",
            [
                os.path.join(FINETUNE_DIR, "train_yolo26s_freeze5", "yolo26s_freeze5.pt"),
                os.path.join(FINETUNE_DIR, "train_yolo26s_freeze5", "railway_freeze5_output", "weights", "best.pt"),
            ],
        ),
        (
            "YOLO26s_freeze10",
            "train_yolo26s_freeze10",
            [
                os.path.join(FINETUNE_DIR, "train_yolo26s_freeze10", "yolo26s_freeze10.pt"),
                os.path.join(FINETUNE_DIR, "train_yolo26s_freeze10", "railway_freeze10_output", "weights", "best.pt"),
                os.path.join(BASE_DIR, "models", "yolo26s_freeze10", "model pytorch", "yolo26s_freeze10.pt"),
                os.path.join(BASE_DIR, "models", "yolo26s_freeze10", "model_pytorch", "yolo26s_freeze10.pt"),
            ],
        ),
    ]

    models = {}
    handled_folders = set()
    for display_name, folder_name, candidates in known_models:
        models[display_name] = first_existing_path(candidates)
        handled_folders.add(folder_name.lower())

    # Tự động quét thêm các thư mục mô hình khác trong research/finetune nếu có
    if os.path.isdir(FINETUNE_DIR):
        for entry in sorted(os.listdir(FINETUNE_DIR)):
            sub_dir = os.path.join(FINETUNE_DIR, entry)
            if not os.path.isdir(sub_dir) or entry.lower() in handled_folders:
                continue
            pt_candidates = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith(".pt")]
            for root, _, files in os.walk(sub_dir):
                if "best.pt" in files:
                    pt_candidates.append(os.path.join(root, "best.pt"))
            if pt_candidates:
                clean_name = entry.replace("train_", "").replace("yolo", "YOLO")
                models[clean_name] = pt_candidates[0]

    return models


def normalize_names(names_obj):
    """Chuẩn hóa trường names (dict hoặc list) sang dạng dict {int: str}"""
    if isinstance(names_obj, dict):
        return {int(k): str(v) for k, v in names_obj.items()}
    if isinstance(names_obj, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names_obj)}
    return {}


def build_class_mapping(model_names, ds_names):
    """
    Xây dựng bảng ánh xạ model_cls_id -> dataset_cls_id (hoặc -1 nếu lớp không có trong dataset).
    Khớp trực tiếp 1-1 theo tên lớp; riêng mô hình COCO 80 lớp ánh xạ thêm 'truck'/'bus' -> 'truck-bus'.
    """
    m_names = normalize_names(model_names)
    d_names = normalize_names(ds_names)
    ds_name_to_id = {name.lower().strip(): cid for cid, name in d_names.items()}

    mapping = {}
    for m_id, m_name in m_names.items():
        norm_name = m_name.lower().strip()
        target_name = COCO_CLASS_MAP.get(norm_name, norm_name)
        mapping[m_id] = ds_name_to_id.get(target_name, -1)

    return mapping


def prepare_dataset_yaml(ds_yaml, temp_dir):
    """
    Kiểm tra đường dẫn val trong data.yaml. Nếu data.yaml dùng đường dẫn tương đối kiểu Roboflow
    (ví dụ `../valid/images`) không tồn tại trên đĩa nhưng thư mục `images/` tồn tại cạnh data.yaml,
    tạo file YAML tạm trỏ đúng vào thư mục ảnh mà không chỉnh sửa file gốc trong dataset_test.
    """
    ds_dir = os.path.abspath(os.path.dirname(ds_yaml))
    with open(ds_yaml, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}

    val_entry = cfg.get("val", "")
    val_resolved = os.path.normpath(os.path.join(ds_dir, str(val_entry))) if val_entry else ""
    local_images = os.path.join(ds_dir, "images")

    if val_resolved and os.path.exists(val_resolved):
        return ds_yaml

    if os.path.isdir(local_images):
        resolved_cfg = dict(cfg)
        resolved_cfg["path"] = ds_dir
        resolved_cfg["train"] = "images"
        resolved_cfg["val"] = "images"
        resolved_cfg["test"] = "images"
        ds_slug = os.path.basename(ds_dir)
        tmp_yaml = os.path.join(temp_dir, f"{ds_slug}_resolved.yaml")
        with open(tmp_yaml, "w", encoding="utf-8") as fp:
            yaml.safe_dump(resolved_cfg, fp, sort_keys=False, allow_unicode=True)
        return tmp_yaml

    return ds_yaml


DATASET_DIR = resolve_dataset_dir()
MODELS = resolve_models()

DATASETS = {
    "roboflow_crossing": os.path.join(DATASET_DIR, "roboflow_crossing", "data.yaml"),
    "rcd_ground": os.path.join(DATASET_DIR, "rcd_ground", "data.yaml"),
    "rcd_elevated": os.path.join(DATASET_DIR, "rcd_elevated", "data.yaml"),
}


class MappedDetectionValidator(DetectionValidator):
    """
    Custom Ultralytics DetectionValidator hỗ trợ ánh xạ chính xác ID lớp giữa từng mô hình
    trong research/ và từng tập dữ liệu kiểm thử trong dataset_test trước khi tính mAP,
    Confusion Matrix và vẽ đồ thị PR/F1.
    """

    def init_metrics(self, model):
        super().init_metrics(model)

        self.model_names = normalize_names(model.names)
        ds_names = normalize_names(self.data.get("names", model.names))
        self.pred_class_map = build_class_mapping(self.model_names, ds_names)

        mapped_valid = [v for v in self.pred_class_map.values() if v >= 0]
        self._has_merged_classes = len(set(mapped_valid)) < len(mapped_valid)

        max_m_id = max(self.model_names.keys()) if self.model_names else -1
        lut = torch.full((max_m_id + 1,), -1, dtype=torch.long)
        for m_id, ds_id in self.pred_class_map.items():
            lut[m_id] = ds_id
        self._class_lut = lut

        # Cập nhật metadata validator & metrics theo đúng hệ lớp của dataset đang đánh giá
        self.names = ds_names
        self.nc = len(ds_names)
        self.class_map = list(range(1, self.nc + 1))
        self.metrics.names = ds_names
        self.confusion_matrix = ConfusionMatrix(
            names=ds_names,
            save_matches=self.args.plots and self.args.visualize,
        )

    def postprocess(self, preds):
        outputs = super().postprocess(preds)
        if not outputs or self._class_lut.numel() == 0:
            return outputs

        remapped_outputs = []
        for pred in outputs:
            cls_tensor = pred["cls"]
            if cls_tensor.numel() == 0:
                remapped_outputs.append(pred)
                continue

            lut = self._class_lut.to(cls_tensor.device)
            raw_cls = cls_tensor.long()
            valid_range = (raw_cls >= 0) & (raw_cls < lut.numel())
            mapped_cls = torch.full_like(raw_cls, -1)
            mapped_cls[valid_range] = lut[raw_cls[valid_range]]

            keep_mask = mapped_cls >= 0
            bboxes = pred["bboxes"][keep_mask]
            conf = pred["conf"][keep_mask]
            mapped_cls = mapped_cls[keep_mask]
            extra = pred["extra"][keep_mask]

            # Chạy class-aware NMS hậu xử lý khi nhiều lớp gộp chung 1 lớp đích
            if self._has_merged_classes and bboxes.shape[0] > 1:
                nms_keep = torchvision.ops.batched_nms(
                    bboxes.float(),
                    conf.float(),
                    mapped_cls,
                    iou_threshold=self.args.iou,
                )
                if self.args.max_det and nms_keep.numel() > self.args.max_det:
                    nms_keep = nms_keep[: self.args.max_det]
                bboxes = bboxes[nms_keep]
                conf = conf[nms_keep]
                mapped_cls = mapped_cls[nms_keep]
                extra = extra[nms_keep]

            remapped_outputs.append({
                "bboxes": bboxes,
                "conf": conf,
                "cls": mapped_cls.to(cls_tensor.dtype),
                "extra": extra,
            })

        return remapped_outputs


def format_class_mapping(model_names, ds_names):
    """Tạo chuỗi mô tả gọn gàng quy tắc ánh xạ lớp giữa mô hình và từng tập dữ liệu"""
    m_names = normalize_names(model_names)
    d_names = normalize_names(ds_names)
    mapping = build_class_mapping(m_names, d_names)
    mapped_lines = []
    skipped_count = 0
    for m_id, m_name in sorted(m_names.items()):
        ds_id = mapping.get(m_id, -1)
        if ds_id >= 0:
            mapped_lines.append(f"      model[{m_id}: {m_name}] -> dataset[{ds_id}: {d_names[ds_id]}]")
        else:
            skipped_count += 1
    if skipped_count > 0:
        mapped_lines.append(f"      (+ lọc bỏ {skipped_count} lớp không có trong data.yaml tập này)")
    return "\n".join(mapped_lines)


def extract_all_per_class_metrics(val_results):
    """
    Trích xuất chính xác Precision, Recall, mAP50, mAP50-95 và số lượng nhãn (Targets)
    cho từng lớp trong tập dữ liệu kiểm thử.
    """
    dataset_classes = normalize_names(val_results.names)
    ap_class_indices = [int(c) for c in getattr(val_results.box, "ap_class_index", [])]
    index_to_pos = {c_idx: pos for pos, c_idx in enumerate(ap_class_indices)}
    nt_per_class = getattr(val_results, "nt_per_class", None)

    per_class = []
    for c_idx, c_name in sorted(dataset_classes.items()):
        targets = int(nt_per_class[c_idx]) if (nt_per_class is not None and c_idx < len(nt_per_class)) else 0
        if c_idx in index_to_pos and targets > 0:
            pos = index_to_pos[c_idx]
            p_c, r_c, ap50_c, ap50_95_c = val_results.box.class_result(pos)
            per_class.append({
                "Class_ID": int(c_idx),
                "Class": str(c_name),
                "Targets": targets,
                "Precision": round(float(p_c), 4),
                "Recall": round(float(r_c), 4),
                "mAP50": round(float(ap50_c), 4),
                "mAP50-95": round(float(ap50_95_c), 4),
            })
        else:
            per_class.append({
                "Class_ID": int(c_idx),
                "Class": str(c_name),
                "Targets": 0,
                "Precision": np.nan,
                "Recall": np.nan,
                "mAP50": np.nan,
                "mAP50-95": np.nan,
            })

    return per_class


def run_evaluation(batch_size=1, imgsz=640, device="cpu", only_models=None):
    """
    Thực thi validation trực tiếp trên dataset_test cho toàn bộ 6 mô hình:
    YOLO26s_Pretrained, YOLO11s, YOLO26s, YOLO26s_freeze3, YOLO26s_freeze5, YOLO26s_freeze10.
    Nếu truyền only_models (danh sách tên model), chỉ chạy đánh giá các model đó và ghép với
    kết quả đã lưu của các model còn lại.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows = []
    class_rows = []
    model_order = list(MODELS.keys())
    target_models = (
        {k: v for k, v in MODELS.items() if k in only_models}
        if only_models
        else MODELS
    )

    print("=" * 102)
    print(f" THỰC NGHIỆM 1: SO SÁNH {len(MODELS)} MÔ HÌNH (PRETRAINED & RESEARCH/) TRÊN DATASET_TEST")
    print(f"[*] Danh sách mô hình : {', '.join(target_models.keys())}")
    print(f"[*] Thư mục dữ liệu   : {DATASET_DIR}")
    print(f"[*] Thiết bị suy luận : {device}")
    print(f"[*] Kích thước ảnh    : {imgsz}x{imgsz}")
    print(f"[*] Kích thước batch  : {batch_size}")
    print("=" * 102)

    with tempfile.TemporaryDirectory(prefix="railsafe_val_yaml_") as tmp_dir:
        resolved_datasets = {}
        for ds_name, ds_yaml in DATASETS.items():
            if os.path.exists(ds_yaml):
                resolved_datasets[ds_name] = prepare_dataset_yaml(ds_yaml, tmp_dir)
            else:
                print(f"[!] Bỏ qua tập dữ liệu {ds_name} (Không tìm thấy {ds_yaml})")

        for model_name, model_path in target_models.items():
            is_builtin_weight = os.path.dirname(model_path) == ""
            if not is_builtin_weight and not os.path.exists(model_path):
                print(f"[!] Cảnh báo: Không tìm thấy file trọng số cho {model_name} tại {model_path}")
                continue

            print(f"\n>>> Đang nạp mô hình {model_name}: {model_path}")
            model = YOLO(model_path)
            print(f"    Hệ lớp gốc của mô hình ({len(model.names)} lớp): {dict(model.names)}")

            for ds_name, eval_yaml in resolved_datasets.items():
                print(f"[*] Đang đánh giá {model_name} trên tập '{ds_name}'...")
                t0 = time.perf_counter()

                val_results = model.val(
                    validator=MappedDetectionValidator,
                    data=eval_yaml,
                    batch=batch_size,
                    imgsz=imgsz,
                    device=device,
                    plots=True,
                    save=True,
                    project=RESULTS_DIR,
                    name=f"{model_name}_{ds_name}",
                    exist_ok=True,
                    verbose=False,
                )
                val_time = time.perf_counter() - t0

                print(f"    Ánh xạ lớp ({ds_name}):")
                print(format_class_mapping(model.names, val_results.names))

                # Lấy các chỉ số tổng hợp
                p = float(val_results.box.mp)
                r = float(val_results.box.mr)
                map50 = float(val_results.box.map50)
                map50_95 = float(val_results.box.map)
                speed_inference = float(val_results.speed.get("inference", 0.0))
                speed_preprocess = float(val_results.speed.get("preprocess", 0.0))
                speed_postprocess = float(val_results.speed.get("postprocess", 0.0))
                total_latency = speed_preprocess + speed_inference + speed_postprocess

                num_targets = int(val_results.nt_per_class.sum()) if getattr(val_results, "nt_per_class", None) is not None else 0

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
                    "Validation_Time_s": round(val_time, 2),
                })

                # Lấy chỉ số chi tiết theo từng lớp
                for c_metrics in extract_all_per_class_metrics(val_results):
                    class_rows.append({
                        "Model": model_name,
                        "Dataset": ds_name,
                        "Class": c_metrics["Class"],
                        "Targets": c_metrics["Targets"],
                        "Precision": c_metrics["Precision"],
                        "Recall": c_metrics["Recall"],
                        "mAP50": c_metrics["mAP50"],
                        "mAP50-95": c_metrics["mAP50-95"],
                    })

                print(
                    f"    [+] Xong {model_name} - {ds_name} ({num_targets} targets): "
                    f"P={p*100:.2f}%, R={r*100:.2f}%, mAP50={map50*100:.2f}%, "
                    f"mAP50-95={map50_95*100:.2f}%, Inference={speed_inference:.2f}ms"
                )

    if not summary_rows:
        print("[!] Không có dữ liệu kết quả nào được ghi nhận. Vui lòng kiểm tra lại đường dẫn!")
        return None, None

    df_summary = pd.DataFrame(summary_rows)
    df_class = pd.DataFrame(class_rows)

    summary_csv = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    class_csv = os.path.join(RESULTS_DIR, "comparison_per_class.csv")

    # Nếu chỉ chạy một số model (only_models), ghép với kết quả đã lưu của các model khác
    if only_models:
        evaluated_set = set(df_summary["Model"].unique())
        if os.path.exists(summary_csv):
            old_sum = pd.read_csv(summary_csv)
            old_sum["Model"] = old_sum["Model"].replace({"YOLO11s_all": "YOLO11s", "YOLO26s_all": "YOLO26s"})
            old_sum = old_sum[~old_sum["Model"].isin(evaluated_set)]
            df_summary = pd.concat([old_sum, df_summary], ignore_index=True)
        if os.path.exists(class_csv):
            old_cls = pd.read_csv(class_csv)
            old_cls["Model"] = old_cls["Model"].replace({"YOLO11s_all": "YOLO11s", "YOLO26s_all": "YOLO26s"})
            old_cls = old_cls[~old_cls["Model"].isin(evaluated_set)]
            df_class = pd.concat([old_cls, df_class], ignore_index=True)

    ds_order = list(DATASETS.keys())
    df_summary["_m_ord"] = df_summary["Model"].apply(lambda x: model_order.index(x) if x in model_order else 999)
    df_summary["_d_ord"] = df_summary["Dataset"].apply(lambda x: ds_order.index(x) if x in ds_order else 999)
    df_summary = df_summary.sort_values(["_m_ord", "_d_ord"]).drop(columns=["_m_ord", "_d_ord"]).reset_index(drop=True)

    df_class["_m_ord"] = df_class["Model"].apply(lambda x: model_order.index(x) if x in model_order else 999)
    df_class["_d_ord"] = df_class["Dataset"].apply(lambda x: ds_order.index(x) if x in ds_order else 999)
    df_class = df_class.sort_values(["_m_ord", "_d_ord", "Class"]).drop(columns=["_m_ord", "_d_ord"]).reset_index(drop=True)

    # Lưu kết quả chi tiết
    df_summary.to_csv(summary_csv, index=False)
    df_class.to_csv(class_csv, index=False)

    # Tính toán bảng chỉ số trung bình trên 3 tập dữ liệu (Macro Average), giữ đúng thứ tự mô hình
    metric_cols = ["Precision", "Recall", "mAP50", "mAP50-95", "Inference_ms", "Total_Latency_ms", "Validation_Time_s"]
    df_avg = df_summary.groupby("Model", as_index=False)[metric_cols].mean().round(4)
    df_avg["_order"] = df_avg["Model"].apply(lambda x: model_order.index(x) if x in model_order else 999)
    df_avg = df_avg.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    avg_csv = os.path.join(RESULTS_DIR, "comparison_average.csv")
    df_avg.to_csv(avg_csv, index=False)

    print("\n" + "=" * 102)
    print(" BẢNG TỔNG HỢP SO SÁNH TRUNG BÌNH 3 TẬP DỮ LIỆU KIỂM THỬ (MACRO AVERAGE METRICS)")
    print("=" * 102)
    print(f"{'Mô hình':<20} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<14} | {'Suy luận (ms)':<15}")
    print("-" * 102)
    for _, row in df_avg.iterrows():
        p_pct = f"{row['Precision']*100:.2f}%"
        r_pct = f"{row['Recall']*100:.2f}%"
        map50_pct = f"{row['mAP50']*100:.2f}%"
        map95_pct = f"{row['mAP50-95']*100:.2f}%"
        lat = f"{row['Inference_ms']:.2f} ms/ảnh"
        print(f"{row['Model']:<20} | {p_pct:<10} | {r_pct:<10} | {map50_pct:<10} | {map95_pct:<14} | {lat:<15}")
    print("=" * 102)

    # Hiển thị bảng trung bình theo từng lớp (chỉ tính trung bình trên các tập có nhãn > 0)
    df_class_valid = df_class[df_class["Targets"] > 0]
    if not df_class_valid.empty:
        df_class_avg = (
            df_class_valid.groupby(["Model", "Class"], as_index=False)
            .agg(
                Total_Targets=("Targets", "sum"),
                Datasets_Present=("Dataset", "count"),
                Precision=("Precision", "mean"),
                Recall=("Recall", "mean"),
                mAP50=("mAP50", "mean"),
                mAP50_95=("mAP50-95", "mean"),
            )
            .round(4)
        )
        df_class_avg["_order"] = df_class_avg["Model"].apply(lambda x: model_order.index(x) if x in model_order else 999)
        df_class_avg = df_class_avg.sort_values(["_order", "Class"]).drop(columns=["_order"]).reset_index(drop=True)
        print("\n BẢNG TRUNG BÌNH THEO TỪNG LỚP ĐỐI TƯỢNG (CHỈ TÍNH TRÊN CÁC TẬP CÓ NHÃN > 0)")
        print("-" * 102)
        print(df_class_avg.to_string(index=False))
        print("=" * 102)

    print(f"[*] Đã lưu bảng tổng kết chi tiết tại : {summary_csv}")
    print(f"[*] Đã lưu bảng chi tiết từng lớp tại : {class_csv}")
    print(f"[*] Đã lưu bảng trung bình tại        : {avg_csv}")

    return df_summary, df_avg


def display_saved_results():
    """Hiển thị nhanh kết quả đã đo đạc nếu file CSV đã tồn tại"""
    avg_csv = os.path.join(RESULTS_DIR, "comparison_average.csv")
    summary_csv = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    class_csv = os.path.join(RESULTS_DIR, "comparison_per_class.csv")

    if not os.path.exists(avg_csv):
        print("[*] Chưa có kết quả đo sẵn, bắt đầu chạy đánh giá trên dataset_test...")
        return run_evaluation()

    df_avg = pd.read_csv(avg_csv)
    print("=" * 102)
    print(" KẾT QUẢ SO SÁNH TRUNG BÌNH ĐÃ LƯU (CÁC MÔ HÌNH TRONG RESEARCH/)")
    print("=" * 102)
    print(df_avg.to_string(index=False))
    if os.path.exists(class_csv):
        df_class = pd.read_csv(class_csv)
        if "Targets" in df_class.columns:
            df_valid = df_class[df_class["Targets"] > 0]
            if not df_valid.empty:
                df_cavg = (
                    df_valid.groupby(["Model", "Class"], as_index=False)
                    .agg(
                        Total_Targets=("Targets", "sum"),
                        Datasets_Present=("Dataset", "count"),
                        Precision=("Precision", "mean"),
                        Recall=("Recall", "mean"),
                        mAP50=("mAP50", "mean"),
                        mAP50_95=("mAP50-95", "mean"),
                    )
                    .round(4)
                )
                print("-" * 102)
                print(" TRUNG BÌNH THEO TỪNG LỚP (TẬP CÓ NHÃN > 0):")
                print(df_cavg.to_string(index=False))
    print("=" * 102)
    return pd.read_csv(summary_csv), df_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh YOLO26s_Pretrained và các mô hình trong research/ trên dataset_test")
    parser.add_argument("--eval", action="store_true", help="Chạy lại toàn bộ quá trình validation trên dataset_test")
    parser.add_argument("--only", type=str, default="", help="Danh sách model cần chạy phân cách bằng dấu phẩy")
    parser.add_argument("--batch", type=int, default=1, help="Kích thước batch (Mặc định 1 để đo chuẩn latency)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị thực thi (cpu, 0, intel:gpu)")
    args = parser.parse_args()

    only_list = [m.strip() for m in args.only.split(",") if m.strip()] if args.only else None

    if args.eval or only_list or not os.path.exists(os.path.join(RESULTS_DIR, "comparison_average.csv")):
        run_evaluation(batch_size=args.batch, imgsz=args.imgsz, device=args.device, only_models=only_list)
    else:
        display_saved_results()
