"""
Thực nghiệm 1: Đánh giá & So sánh Đối đầu 3 Kiến trúc Mô hình Nhận diện (YOLO11s vs YOLO26s vs YOLO26m)
So sánh 3 mô hình huấn luyện trên dữ liệu an toàn đường sắt đầy đủ 5 lớp
(0: car, 1: motorcycle, 2: person, 3: truck-bus, 4: train):
1. YOLO11s (Baseline)          - research/finetune/train_yolo11s/yolo11s.pt
2. YOLO26s (Edge Real-Time)    - models/yolo26s/model_pytorch/train-railway-yolo26s-all.pt
3. YOLO26m (High-Accuracy GPU) - models/yolo26m/model_pytorch/yolo26m.pt

Đánh giá chéo độc lập trên 3 tập dữ liệu kiểm thử trong dataset_test/ (404 ảnh / 1.238 đối tượng):
- dataset_test/roboflow_crossing/data.yaml (119 ảnh, 5 lớp)
- dataset_test/rcd_ground/data.yaml        (185 ảnh, 3 lớp)
- dataset_test/rcd_elevated/data.yaml      (100 ảnh, 4 lớp)

Cơ chế tự động đồng bộ khi đánh giá (MappedDetectionValidator):
- Tự động xử lý đường dẫn Roboflow (`../valid/images` -> `images`) qua YAML tạm, giữ nguyên `dataset_test/`.
- Ánh xạ chính xác ID lớp 1-1 theo tên lớp giữa mô hình và từng tập dữ liệu kiểm thử.
- Xuất đầy đủ bảng CSV (specs, summary, average, per_class, per_class_avg) và biểu đồ so sánh trực quan.
"""

import os
import sys
import time
import tempfile
import argparse
import yaml
import numpy as np
import pandas as pd

# Đảm bảo xuất tiếng Việt ổn định trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Thiết lập đường dẫn cơ sở
EXP1_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(EXP1_DIR, "..", ".."))
RESULTS_DIR = os.path.join(EXP1_DIR, "results")
FINETUNE_DIR = os.path.join(BASE_DIR, "research", "finetune")

# Đảm bảo Ultralytics luôn có thư mục cấu hình ghi được
if not os.environ.get("YOLO_CONFIG_DIR"):
    _tmp_cfg = os.path.join(tempfile.gettempdir(), "railsafe_ultralytics_cfg")
    try:
        os.makedirs(_tmp_cfg, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = _tmp_cfg
    except Exception:
        _local_cfg = os.path.join(EXP1_DIR, ".ultralytics_cfg")
        os.makedirs(_local_cfg, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = _local_cfg

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils.torch_utils import get_flops, get_num_params


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
    """Tự động dò tìm đường dẫn trọng số hợp lệ cho 3 mô hình YOLO11s, YOLO26s và YOLO26m"""
    return {
        "YOLO11s": first_existing_path([
            os.path.join(FINETUNE_DIR, "train_yolo11s", "yolo11s.pt"),
            os.path.join(FINETUNE_DIR, "train_yolo11s", "train-railway-yolo11s", "weights", "best.pt"),
            os.path.join(BASE_DIR, "models", "yolo11s", "model_pytorch", "train-railway-yolo11s-all.pt"),
        ]),
        "YOLO26s": first_existing_path([
            os.path.join(BASE_DIR, "models", "yolo26s", "model_pytorch", "train-railway-yolo26s-all.pt"),
            os.path.join(FINETUNE_DIR, "train_yolo26s", "yolo26s.pt"),
            os.path.join(FINETUNE_DIR, "train_yolo26s", "train-railway-yolo26s", "weights", "best.pt"),
        ]),
        "YOLO26m": first_existing_path([
            os.path.join(BASE_DIR, "models", "yolo26m", "model_pytorch", "yolo26m.pt"),
            os.path.join(FINETUNE_DIR, "train_yolo26m", "yolo26m.pt"),
            os.path.join(FINETUNE_DIR, "train_yolo26m", "model", "yolo26m.pt"),
            os.path.join(BASE_DIR, "Fine-tune-Dataset-Lam", "Fine-tune-Dataset-Lam", "model", "best.pt"),
        ]),
    }


def normalize_names(names_obj):
    if isinstance(names_obj, dict):
        return {int(k): str(v) for k, v in names_obj.items()}
    if isinstance(names_obj, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names_obj)}
    return {}


def build_class_mapping(model_names, ds_names):
    m_names = normalize_names(model_names)
    d_names = normalize_names(ds_names)
    ds_name_to_id = {name.lower().strip(): cid for cid, name in d_names.items()}
    return {
        m_id: ds_name_to_id.get(m_name.lower().strip(), -1)
        for m_id, m_name in m_names.items()
    }


def prepare_dataset_yaml(ds_yaml, temp_dir):
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
    """Custom Ultralytics DetectionValidator ánh xạ 1-1 ID lớp theo tên lớp giữa mô hình và tập kiểm thử."""

    def init_metrics(self, model):
        super().init_metrics(model)

        self.model_names = normalize_names(model.names)
        ds_names = normalize_names(self.data.get("names", model.names))
        self.pred_class_map = build_class_mapping(self.model_names, ds_names)

        max_m_id = max(self.model_names.keys()) if self.model_names else -1
        lut = torch.full((max_m_id + 1,), -1, dtype=torch.long)
        for m_id, ds_id in self.pred_class_map.items():
            lut[m_id] = ds_id
        self._class_lut = lut

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
            remapped_outputs.append({
                "bboxes": pred["bboxes"][keep_mask],
                "conf": pred["conf"][keep_mask],
                "cls": mapped_cls[keep_mask].to(cls_tensor.dtype),
                "extra": pred["extra"][keep_mask],
            })

        return remapped_outputs


def inspect_model_specs(model_name, model_path, model_obj):
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    n_layers = len(list(model_obj.model.modules()))
    n_params = get_num_params(model_obj.model)
    try:
        gflops = float(get_flops(model_obj.model, imgsz=640))
    except Exception:
        gflops = 0.0
    try:
        info = model_obj.model.info(detailed=False, verbose=False)
        if isinstance(info, (tuple, list)) and len(info) >= 4:
            n_layers, n_params, _, gflops = info[:4]
    except Exception:
        pass
    return {
        "Model": model_name,
        "Path": os.path.relpath(model_path, BASE_DIR),
        "File_Size_MB": round(file_size_mb, 2),
        "Layers": int(n_layers),
        "Params_M": round(float(n_params) / 1e6, 2),
        "GFLOPs": round(float(gflops), 2),
        "Num_Classes": len(model_obj.names),
    }


def format_class_mapping(model_names, ds_names):
    m_names = normalize_names(model_names)
    d_names = normalize_names(ds_names)
    mapping = build_class_mapping(m_names, d_names)
    lines = []
    for m_id, m_name in sorted(m_names.items()):
        ds_id = mapping.get(m_id, -1)
        if ds_id >= 0:
            lines.append(f"      model[{m_id}: {m_name}] -> dataset[{ds_id}: {d_names[ds_id]}]")
        else:
            lines.append(f"      model[{m_id}: {m_name}] -> (bỏ qua: không có trong data.yaml tập này)")
    return "\n".join(lines)


def extract_all_per_class_metrics(val_results):
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
            f1_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
            per_class.append({
                "Class_ID": int(c_idx),
                "Class": str(c_name),
                "Targets": targets,
                "Precision": round(float(p_c), 4),
                "Recall": round(float(r_c), 4),
                "F1": round(float(f1_c), 4),
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
                "F1": np.nan,
                "mAP50": np.nan,
                "mAP50-95": np.nan,
            })
    return per_class


def generate_comparison_charts(df_avg, df_summary, df_class_avg, df_specs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[!] Bỏ qua vẽ biểu đồ (không có matplotlib): {e}")
        return

    colors = {"YOLO11s": "#64748b", "YOLO26s": "#0d828a", "YOLO26m": "#e11d48"}
    models = df_avg["Model"].tolist()
    n_m = len(models)
    width = 0.8 / max(1, n_m)

    # 1. Macro Average Metrics & Per-Dataset mAP@0.5
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    metrics = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]
    x = np.arange(len(metrics))
    for i, m in enumerate(models):
        row = df_avg[df_avg["Model"] == m].iloc[0]
        vals = [row[k] * 100 for k in metrics]
        offset = (i - (n_m - 1) / 2) * width
        bars = axes[0].bar(x + offset, vals, width, label=m, color=colors.get(m))
        for b in bars:
            h = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_title("Macro Average Metrics trên 3 tập dataset_test (404 ảnh)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Precision", "Recall", "F1-Score", "mAP@0.5", "mAP@50-95"])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Phần trăm (%)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()

    ds_list = list(DATASETS.keys())
    x2 = np.arange(len(ds_list))
    for i, m in enumerate(models):
        vals = []
        for ds in ds_list:
            sub = df_summary[(df_summary["Model"] == m) & (df_summary["Dataset"] == ds)]
            vals.append(sub.iloc[0]["mAP50"] * 100 if not sub.empty else 0.0)
        offset = (i - (n_m - 1) / 2) * width
        bars = axes[1].bar(x2 + offset, vals, width, label=m, color=colors.get(m))
        for b in bars:
            h = b.get_height()
            axes[1].text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_title("mAP@0.5 theo từng tập kiểm thử độc lập")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(ds_list)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("mAP@0.5 (%)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "chart_3models_accuracy.png"), dpi=160)
    plt.close(fig)

    # 2. Per-Class mAP@0.5 (5 classes)
    if not df_class_avg.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        classes = sorted(df_class_avg["Class"].unique().tolist())
        xc = np.arange(len(classes))
        for i, m in enumerate(models):
            vals = []
            for cls in classes:
                sub = df_class_avg[(df_class_avg["Model"] == m) & (df_class_avg["Class"] == cls)]
                vals.append(sub.iloc[0]["mAP50"] * 100 if not sub.empty else 0.0)
            offset = (i - (n_m - 1) / 2) * width
            bars = ax.bar(xc + offset, vals, width, label=m, color=colors.get(m))
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_title("So sánh mAP@0.5 trung bình theo từng lớp (5 lớp)")
        ax.set_xticks(xc)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 108)
        ax.set_ylabel("mAP@0.5 (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "chart_3models_per_class.png"), dpi=160)
        plt.close(fig)

    # 3. Pareto Trade-off: Latency & Complexity vs mAP
    if not df_specs.empty:
        merged = pd.merge(df_specs, df_avg[["Model", "mAP50", "mAP50-95", "Inference_ms"]], on="Model")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        bar_colors = [colors.get(m, "#475569") for m in merged["Model"]]

        b1 = axes[0].bar(merged["Model"], merged["Inference_ms"], color=bar_colors, width=0.5)
        for b in b1:
            h = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, h + 3, f"{h:.1f} ms", ha="center", va="bottom", fontsize=8.5)
        axes[0].set_title("Độ trễ suy luận TB trên CPU (ms/ảnh — thấp hơn tốt hơn)")
        axes[0].set_ylabel("Inference Latency (ms)")
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)

        b2 = axes[1].bar(merged["Model"], merged["GFLOPs"], color=bar_colors, width=0.5)
        for b, (_, r) in zip(b2, merged.iterrows()):
            h = b.get_height()
            axes[1].text(
                b.get_x() + b.get_width() / 2, h + 1.2,
                f"{h:.1f}G\n({r['File_Size_MB']:.1f} MB | {r['Params_M']:.2f}M)",
                ha="center", va="bottom", fontsize=8
            )
        axes[1].set_title("Độ phức tạp tính toán (GFLOPs), Tham số & Dung lượng")
        axes[1].set_ylabel("GFLOPs")
        axes[1].set_ylim(0, max(merged["GFLOPs"]) * 1.28)
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "chart_3models_complexity.png"), dpi=160)
        plt.close(fig)


def run_evaluation(batch_size=1, imgsz=640, device="cpu"):
    """Thực thi validation trên dataset_test cho cả 3 mô hình YOLO11s, YOLO26s và YOLO26m."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    spec_rows = []
    summary_rows = []
    class_rows = []
    model_order = list(MODELS.keys())

    print("=" * 98)
    print(" THỰC NGHIỆM 1: ĐÁNH GIÁ ĐỐI ĐẦU 3 MÔ HÌNH YOLO11s VS YOLO26s VS YOLO26m TRÊN DATASET_TEST")
    print(f"[*] Thư mục dữ liệu   : {DATASET_DIR}")
    print(f"[*] Thiết bị suy luận : {device}")
    print(f"[*] Kích thước ảnh    : {imgsz}x{imgsz} | batch={batch_size}")
    print("=" * 98)

    with tempfile.TemporaryDirectory(prefix="railsafe_val_yaml_") as tmp_dir:
        resolved_datasets = {}
        for ds_name, ds_yaml in DATASETS.items():
            if os.path.exists(ds_yaml):
                resolved_datasets[ds_name] = prepare_dataset_yaml(ds_yaml, tmp_dir)
            else:
                print(f"[!] Bỏ qua tập dữ liệu {ds_name} (Không tìm thấy {ds_yaml})")

        for model_name, model_path in MODELS.items():
            if not os.path.exists(model_path):
                print(f"[!] Cảnh báo: Không tìm thấy file trọng số cho {model_name} tại {model_path}")
                continue

            print(f"\n>>> Đang nạp mô hình {model_name}: {model_path}")
            model = YOLO(model_path)
            specs = inspect_model_specs(model_name, model_path, model)
            spec_rows.append(specs)
            print(
                f"    Thông số: {specs['File_Size_MB']} MB | {specs['Layers']} layers | "
                f"{specs['Params_M']}M params | {specs['GFLOPs']} GFLOPs | classes={dict(model.names)}"
            )

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
                    save=False,
                    project=RESULTS_DIR,
                    name=f"{model_name}_{ds_name}",
                    exist_ok=True,
                    verbose=False,
                )
                val_time = time.perf_counter() - t0

                print(f"    Ánh xạ lớp ({ds_name}):")
                print(format_class_mapping(model.names, val_results.names))

                p = float(val_results.box.mp)
                r = float(val_results.box.mr)
                f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                map50 = float(val_results.box.map50)
                map50_95 = float(val_results.box.map)
                speed_inference = float(val_results.speed.get("inference", 0.0))
                speed_preprocess = float(val_results.speed.get("preprocess", 0.0))
                speed_postprocess = float(val_results.speed.get("postprocess", 0.0))
                total_latency = speed_preprocess + speed_inference + speed_postprocess
                num_targets = (
                    int(val_results.nt_per_class.sum())
                    if getattr(val_results, "nt_per_class", None) is not None
                    else 0
                )

                summary_rows.append({
                    "Model": model_name,
                    "Dataset": ds_name,
                    "Targets": num_targets,
                    "Precision": round(p, 4),
                    "Recall": round(r, 4),
                    "F1": round(f1, 4),
                    "mAP50": round(map50, 4),
                    "mAP50-95": round(map50_95, 4),
                    "Inference_ms": round(speed_inference, 2),
                    "Total_Latency_ms": round(total_latency, 2),
                    "Validation_Time_s": round(val_time, 2),
                })

                for c_metrics in extract_all_per_class_metrics(val_results):
                    class_rows.append({
                        "Model": model_name,
                        "Dataset": ds_name,
                        "Class": c_metrics["Class"],
                        "Targets": c_metrics["Targets"],
                        "Precision": c_metrics["Precision"],
                        "Recall": c_metrics["Recall"],
                        "F1": c_metrics["F1"],
                        "mAP50": c_metrics["mAP50"],
                        "mAP50-95": c_metrics["mAP50-95"],
                    })

                print(
                    f"    [+] Xong {model_name} - {ds_name} ({num_targets} targets): "
                    f"P={p*100:.2f}%, R={r*100:.2f}%, F1={f1*100:.2f}%, "
                    f"mAP50={map50*100:.2f}%, mAP50-95={map50_95*100:.2f}%, Inf={speed_inference:.2f}ms"
                )

    if not summary_rows:
        print("[!] Không có dữ liệu kết quả nào được ghi nhận. Vui lòng kiểm tra lại đường dẫn!")
        return None, None

    df_specs = pd.DataFrame(spec_rows)
    df_summary = pd.DataFrame(summary_rows)
    df_class = pd.DataFrame(class_rows)

    metric_cols = ["Precision", "Recall", "F1", "mAP50", "mAP50-95", "Inference_ms", "Total_Latency_ms", "Validation_Time_s"]
    df_avg = df_summary.groupby("Model", as_index=False)[metric_cols].mean().round(4)
    df_avg["_order"] = df_avg["Model"].apply(lambda x: model_order.index(x) if x in model_order else 999)
    df_avg = df_avg.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    df_class_valid = df_class[df_class["Targets"] > 0]
    df_class_avg = pd.DataFrame()
    if not df_class_valid.empty:
        df_class_avg = (
            df_class_valid.groupby(["Model", "Class"], as_index=False)
            .agg(
                Total_Targets=("Targets", "sum"),
                Datasets_Present=("Dataset", "count"),
                Precision=("Precision", "mean"),
                Recall=("Recall", "mean"),
                F1=("F1", "mean"),
                mAP50=("mAP50", "mean"),
                mAP50_95=("mAP50-95", "mean"),
            )
            .round(4)
        )
        df_class_avg["_order"] = df_class_avg["Model"].apply(lambda x: model_order.index(x) if x in model_order else 999)
        df_class_avg = df_class_avg.sort_values(["_order", "Class"]).drop(columns=["_order"]).reset_index(drop=True)

    df_specs.to_csv(os.path.join(RESULTS_DIR, "comparison_model_specs.csv"), index=False)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "comparison_summary.csv"), index=False)
    df_class.to_csv(os.path.join(RESULTS_DIR, "comparison_per_class.csv"), index=False)
    df_avg.to_csv(os.path.join(RESULTS_DIR, "comparison_average.csv"), index=False)
    if not df_class_avg.empty:
        df_class_avg.to_csv(os.path.join(RESULTS_DIR, "comparison_per_class_avg.csv"), index=False)

    generate_comparison_charts(df_avg, df_summary, df_class_avg, df_specs)

    print("\n" + "=" * 98)
    print(" BẢNG TỔNG HỢP SO SÁNH TRUNG BÌNH 3 TẬP DỮ LIỆU KIỂM THỬ (MACRO AVERAGE METRICS)")
    print("=" * 98)
    print(f"{'Mô hình':<10} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'mAP@0.5':<9} | {'mAP50-95':<9} | {'Suy luận (ms)':<14}")
    print("-" * 98)
    for _, row in df_avg.iterrows():
        print(
            f"{row['Model']:<10} | {row['Precision']*100:>8.2f}% | {row['Recall']*100:>8.2f}% | "
            f"{row['F1']*100:>8.2f}% | {row['mAP50']*100:>8.2f}% | {row['mAP50-95']*100:>8.2f}% | "
            f"{row['Inference_ms']:>8.2f} ms"
        )
    print("=" * 98)

    if not df_class_avg.empty:
        print("\n BẢNG TRUNG BÌNH THEO TỪNG LỚP ĐỐI TƯỢNG (5 LỚP)")
        print("-" * 98)
        print(df_class_avg.to_string(index=False))
        print("=" * 98)

    return df_summary, df_avg


def display_saved_results():
    avg_csv = os.path.join(RESULTS_DIR, "comparison_average.csv")
    summary_csv = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    class_avg_csv = os.path.join(RESULTS_DIR, "comparison_per_class_avg.csv")

    if not os.path.exists(avg_csv):
        print("[*] Chưa có kết quả đo sẵn, bắt đầu chạy đánh giá trên dataset_test...")
        return run_evaluation()

    df_avg = pd.read_csv(avg_csv)
    print("=" * 98)
    print(" KẾT QUẢ SO SÁNH TRUNG BÌNH ĐÃ LƯU (YOLO11s VS YOLO26s VS YOLO26m)")
    print("=" * 98)
    print(df_avg.to_string(index=False))
    if os.path.exists(class_avg_csv):
        print("-" * 98)
        print(" TRUNG BÌNH THEO TỪNG LỚP (5 LỚP):")
        print(pd.read_csv(class_avg_csv).to_string(index=False))
    print("=" * 98)
    return pd.read_csv(summary_csv), df_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh 3 mô hình YOLO11s vs YOLO26s vs YOLO26m trên dataset_test")
    parser.add_argument("--eval", action="store_true", help="Chạy lại toàn bộ quá trình validation trên dataset_test")
    parser.add_argument("--batch", type=int, default=1, help="Kích thước batch (Mặc định 1 để đo chuẩn latency)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị thực thi (cpu, 0, intel:gpu)")
    args = parser.parse_args()

    if args.eval or not os.path.exists(os.path.join(RESULTS_DIR, "comparison_average.csv")):
        run_evaluation(batch_size=args.batch, imgsz=args.imgsz, device=args.device)
    else:
        display_saved_results()
