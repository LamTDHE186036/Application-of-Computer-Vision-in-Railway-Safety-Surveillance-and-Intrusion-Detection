"""Draw the two detection confusion matrices from the saved 119-image test predictions."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


from image_metrics import fixed, overlaps


def confusion_matrix(rows, names, confidence, iou_threshold, reference):
    """Keep same-class matches from the fixed-threshold evaluation."""
    totals, audits = fixed(rows, names, threshold=confidence)
    if totals != reference:
        raise ValueError("The saved predictions do not reproduce the reported metrics")

    background = len(names)
    matrix = np.zeros((background + 1, background + 1), dtype=int)
    for row, audit in zip(rows, audits, strict=True):
        truth = row["ground_truth"]
        matched_truth = set()
        extra_predictions = []
        for prediction in audit["predictions"]:
            if prediction["matched"]:
                matched_truth.add(prediction["gt_index"])
                matrix[prediction["cls"], prediction["cls"]] += 1
            else:
                extra_predictions.append(prediction)

        # Pair the remaining boxes of different classes by descending IoU.
        remaining_truth = [
            (i, box) for i, box in enumerate(truth) if i not in matched_truth
        ]
        candidates = []
        for prediction_index, prediction in enumerate(extra_predictions):
            ious = overlaps(prediction["xyxy"], [box["xyxy"] for _, box in remaining_truth])
            for (truth_index, box), iou in zip(remaining_truth, ious, strict=True):
                if box["cls"] != prediction["cls"] and iou >= iou_threshold:
                    candidates.append((-iou, prediction_index, truth_index))

        paired_predictions = set()
        for _, prediction_index, truth_index in sorted(candidates):
            if truth_index in matched_truth or prediction_index in paired_predictions:
                continue
            matched_truth.add(truth_index)
            paired_predictions.add(prediction_index)
            matrix[truth[truth_index]["cls"], extra_predictions[prediction_index]["cls"]] += 1

        for truth_index, box in remaining_truth:
            if truth_index not in matched_truth:
                matrix[box["cls"], background] += 1
        for prediction_index, prediction in enumerate(extra_predictions):
            if prediction_index not in paired_predictions:
                matrix[background, prediction["cls"]] += 1

    for i, name in enumerate(names):
        counts = reference[name]
        observed = (matrix[i, i], matrix[:, i].sum() - matrix[i, i],
                    matrix[i, :].sum() - matrix[i, i])
        if observed != (counts["tp"], counts["fp"], counts["fn"]):
            raise ValueError(f"Matrix totals do not match TP, FP and FN for {name}")
    return matrix


def draw_matrices(output_path, matrices, labels, confidence, iou_threshold):
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.15))
    fig.subplots_adjust(left=0.135, right=0.93, bottom=0.265, top=0.8, wspace=0.32)
    fig.suptitle("Detection confusion matrices", fontsize=21, fontweight="bold", y=0.975)
    fig.text(0.5, 0.891, f"119 test images  |  Confidence ≥ {confidence:.2f}  |  IoU ≥ {iou_threshold:.2f}",
             ha="center", fontsize=12, color="#364152")
    legend_axis = fig.add_axes([0.94, 0.283, 0.012, 0.50])
    color_max = max(int(matrix.max()) for matrix in matrices.values())

    for axis, (title, matrix) in zip(axes, matrices.items(), strict=True):
        colors = np.ma.array(matrix, mask=False)
        colors.mask[-1, -1] = True
        image = axis.imshow(colors, cmap=plt.colormaps["Blues"].with_extremes(bad="#e9edf2"),
                            vmin=0, vmax=color_max)
        axis.set_title(title, loc="center", fontsize=15, fontweight="bold", pad=10)
        axis.set_xticks(range(len(labels)), labels, rotation=38, ha="right",
                        rotation_mode="anchor", fontsize=11)
        axis.set_yticks(range(len(labels)), labels, fontsize=11)
        axis.set_xlabel("Predicted class", fontsize=13, labelpad=7)
        axis.set_ylabel("Ground-truth class", fontsize=13, labelpad=8)
        axis.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        axis.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        axis.grid(which="minor", color="#cad5e2", linewidth=0.65)
        axis.tick_params(axis="both", which="both", length=0, pad=5)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for (row, column), count in np.ndenumerate(matrix):
            value = "—" if colors.mask[row, column] else str(count)
            color = "white" if count > color_max * 0.70 else "#192b45"
            if value == "0":
                color = "#536276"
            axis.text(column, row, value, ha="center", va="center", fontsize=12.5,
                      fontweight="bold" if row == column else "normal", color=color)

    legend = fig.colorbar(image, cax=legend_axis)
    legend.set_ticks([0, 40, 80, 120, 160])
    legend.ax.tick_params(labelsize=10)
    legend.outline.set_visible(False)
    legend.ax.set_title("Count", fontsize=10, pad=5)
    fig.text(0.5, 0.03, "Background row: unpaired predictions.  Background column: unpaired ground-truth boxes.",
             ha="center", fontsize=11, color="#364152")
    fig.savefig(output_path, dpi=320, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Output directory produced by score.py.")
    run_dir = parser.parse_args().run_dir
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    names = protocol["classes"]
    labels = ["Car", "Motorcycle", "Person", "Truck / bus", "Train", "Background"]
    confidence, iou_threshold = protocol["fixed_confidence"], protocol["fixed_iou"]
    if names != ["car", "motorcycle", "person", "truck-bus", "train"] or iou_threshold != 0.5:
        raise ValueError("Expected the five thesis classes and the fixed evaluator's IoU of 0.50")

    matrices = {}
    for key, title in (("coco", "YOLO26m pretrained"), ("epoch13", "YOLO26m fine-tuned")):
        rows = json.loads((run_dir / f"{key}-predictions.json").read_text(encoding="utf-8"))
        expected = metrics[key]["by_source"]["roboflow_crossing"]
        if len(rows) != 119 or any(row["group"] != "roboflow_crossing" for row in rows):
            raise ValueError("Expected the 119-image thesis test set")
        matrices[title] = confusion_matrix(rows, names, confidence, iou_threshold, expected["fixed"])

    draw_matrices(run_dir / "confusion-matrix-test-119.png", matrices, labels, confidence, iou_threshold)
    output = {
        "images": 119,
        "confidence_threshold": confidence,
        "iou_threshold": iou_threshold,
        "rows_ground_truth_columns_prediction": labels,
        "background_background": None,
        "matching": "Same-class matches from the fixed evaluator, then one-to-one different-class matches by descending IoU.",
        "matrices": {title: matrix.tolist() for title, matrix in matrices.items()},
    }
    (run_dir / "confusion-matrices.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
