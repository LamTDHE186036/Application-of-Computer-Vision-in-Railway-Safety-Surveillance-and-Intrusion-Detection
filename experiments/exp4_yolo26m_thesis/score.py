"""Run the two locked YOLO26m checkpoints on the original Dataset_test image annotations."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("coco", type=Path)
    parser.add_argument("epoch13", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", help="Evaluate one source directory inside the dataset root.")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output / ".settings"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    import cv2
    import yaml
    import numpy as np
    import torch
    from torch import nn
    from ultralytics import YOLO, __version__
    from image_metrics import ap, fixed

    if __version__ != "8.4.161":
        raise ValueError("Use the archived evaluation version ultralytics==8.4.161")
    torch.set_num_threads(4)
    names = ("car", "motorcycle", "person", "truck-bus", "train")
    coco_categories = (("car",), ("motorcycle",), ("person",), ("truck", "bus"), ("train",))
    manifest_path = args.output / "manifest.json"
    manifest = []
    source_names = {}
    sources = sorted(
        source for source in args.dataset.iterdir()
        if source.is_dir() and (args.source is None or source.name == args.source)
    )
    if not sources:
        parser.error("No matching source directories found in the dataset root.")
    for source in sources:
        yaml_path = source / "data.yaml"
        native_names = yaml.safe_load(yaml_path.read_bytes())["names"]
        if isinstance(native_names, dict):
            native_names = [native_names[i] for i in range(len(native_names))]
        source_names[source.name] = native_names
        covered = [names.index(label) for label in native_names]
        for img in sorted((source / "images").iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                continue
            label = source / "labels" / (img.stem + ".txt")
            manifest.append({
                "file": img.relative_to(args.dataset).as_posix(),
                "label": label.relative_to(args.dataset).as_posix(),
                "sha256": hash_file(img), "label_sha256": hash_file(label),
                "group": source.name, "covered_ids": covered,
            })
    save_json(manifest_path, manifest)
    settings = dict(imgsz=640, conf=.001, device=args.device, quantize=32, nms=False,
                    max_det=300, augment=False, rect=True, verbose=False)
    checkpoints = {"coco": args.coco, "epoch13": args.epoch13}
    model_hashes = {tag: hash_file(path) for tag, path in checkpoints.items()}
    expected_hashes = {
        "coco": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
        "epoch13": "5146bf56882d88a9216517c03ee07a238aba06f59ff4fbea8d90917653967bfb",
    }
    if model_hashes != expected_hashes:
        raise ValueError("The checkpoint hashes do not match the thesis models")

    class ProjectedClassifier(nn.Module):
        def __init__(self, original, groups):
            super().__init__()
            self.original = original
            self.groups = groups

        def forward(self, features):
            logits = self.original(features)
            return torch.cat([logits[:, indices].amax(dim=1, keepdim=True) for indices in self.groups], dim=1)

    def project_coco(model):
        if len(model.names) != 80:
            raise ValueError("The baseline must start with the original COCO checkpoint")
        index_by_name = {label: idx for idx, label in model.names.items()}
        groups = [tuple(index_by_name[label] for label in group) for group in coco_categories]
        head = model.model[-1]
        for branch in (head.cv3, head.one2one_cv3):
            for scale in branch:
                scale[-1] = ProjectedClassifier(scale[-1], groups)
        head.nc = model.nc = model.yaml["nc"] = len(names)
        head.no = head.nc + 4 * head.reg_max
        model.names = dict(enumerate(names))

    def load_truth(row, width, height):
        label_file = args.dataset / row["label"]
        if hash_file(label_file) != row["label_sha256"]:
            raise ValueError(f"Label file differs from manifest: {row['label']}")
        labels = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            cls, cx, cy, bw, bh = map(float, line.split())
            box = [(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height]
            if not cls.is_integer() or not 0 <= cls < len(row["covered_ids"]):
                raise ValueError(f"Class ID outside the source taxonomy: {label_file}")
            if not all(map(math.isfinite, box)) or bw <= 0 or bh <= 0:
                raise ValueError(f"Invalid bounding box: {label_file}")
            mapped = row["covered_ids"][int(cls)]
            labels.append(dict(cls=mapped, xyxy=box, ap_xyxy=box))
        return labels

    def score(rows):
        counts = {name: sum(i in row["covered_ids"] for row in rows) for i, name in enumerate(names)}
        return dict(images=len(rows), evaluated_images_per_class=counts,
                    fixed=fixed(rows, names, threshold=.35)[0], **ap(rows, names))

    output = {}
    for tag, checkpoint in checkpoints.items():
        model = YOLO(str(checkpoint))
        if tag == "coco":
            project_coco(model.model)
        if tuple(model.names.values()) != names:
            raise ValueError("Checkpoint and evaluation classes are not aligned")
        rows = []
        for row in manifest:
            source = args.dataset / row["file"]
            if hash_file(source) != row["sha256"]:
                raise ValueError(f"Image differs from manifest: {row['file']}")
            image = cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            truth = load_truth(row, width, height)
            result = model.predict(image, **settings)[0]
            predictions = []
            for x1, y1, x2, y2, confidence, cls in result.boxes.data.cpu().tolist():
                class_id = int(cls)
                if x1 < x2 and y1 < y2 and class_id in row["covered_ids"]:
                    box = [x1, y1, x2, y2]
                    predictions.append(dict(cls=class_id, conf=confidence, xyxy=box, ap_xyxy=box))
            rows.append(dict(file=row["file"], group=row["group"], covered_ids=row["covered_ids"],
                             ground_truth=truth, predictions=predictions))
        groups = sorted({row["group"] for row in rows})
        output[tag] = {
            "all_coverage": score(rows),
            "by_source": {group: score([row for row in rows if row["group"] == group]) for group in groups},
        }
        save_json(args.output / f"{tag}-predictions.json", rows)
        for group, metrics in output[tag]["by_source"].items():
            print(f"{tag} {group}: mAP50={metrics['map50']:.4f}, "
                  f"mAP50–95={metrics['map50_95']:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(args.output / "metrics.json", output)
    save_json(args.output / "protocol.json", {
        "classes": names, "settings": settings, "checkpoints_sha256": model_hashes,
        "image_manifest_sha256": hash_file(manifest_path), "baseline_projection": coco_categories,
        "fixed_confidence": .35, "fixed_iou": .50,
        "source_names": source_names,
        "coverage_rule": "A source contributes ground truth and predictions only for the classes declared in its data.yaml.",
        "ultralytics": __version__, "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if args.device != "cpu" else "CPU",
    })


if __name__ == "__main__":
    main()
