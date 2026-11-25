# THỰC NGHIỆM 1: ĐÁNH GIÁ & SO SÁNH ĐỐI ĐẦU 3 KIẾN TRÚC MÔ HÌNH NHẬN DIỆN (YOLO11s VS YOLO26s VS YOLO26m)

## 1. Mục tiêu Thực nghiệm & Cấu hình Đánh giá

Thực nghiệm đánh giá so sánh trực tiếp năng lực nhận diện chướng ngại vật xâm nhập hành lang an toàn đường sắt giữa **3 kiến trúc mô hình** 5 lớp (`0: car`, `1: motorcycle`, `2: person`, `3: truck-bus`, `4: train` — gồm 4 lớp chướng ngại vật xâm phạm và 1 lớp đoàn tàu hỏa):

| Mô hình | Vai trò trong Hệ thống | Đường dẫn trọng số | Dung lượng | Params (Raw / Fused) | GFLOPs (Raw / Fused) | Cấu hình huấn luyện |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`YOLO11s`** | Baseline đối chứng | `research/finetune/train_yolo11s/yolo11s.pt` | **18.29 MB** | **9.43M** / 9.41M | **21.68** / 21.4 | `Dataset_all` (2,125 train / 532 val), 50 epochs, batch=16, imgsz=640, seed=42 |
| **`YOLO26s`** | **Triển khai Edge Real-Time (Intel iGPU/CPU)** | `models/yolo26s/model_pytorch/train-railway-yolo26s-all.pt` | 19.38 MB | 9.95M / 9.47M | 22.76 / 20.8 | `Dataset_all` (2,125 train / 532 val), 50 epochs, batch=16, imgsz=640, seed=42 |
| **`YOLO26m`** | **Triển khai High-Accuracy (NVIDIA CUDA GPU)** | `models/yolo26m/model_pytorch/yolo26m.pt` | 42.03 MB | 21.78M / 20.35M | 74.99 / 68.1 | `Dataset_Lam` (2,100 train / 532 val), checkpoint epoch 13/20, freeze=6, AdamW lr0=1e-5, COCO class-remap |

### Cơ chế đồng bộ đường dẫn & ánh xạ lớp (`MappedDetectionValidator`)
Bộ kiểm thử độc lập `dataset_test/` đã chuẩn hóa đầy đủ 5 lớp tương thích với mô hình (`car`, `motorcycle`, `person`, `train`, `truck-bus`), nhưng thứ tự ID và tập lớp có mặt ở mỗi tập con khác nhau (do xuất từ Roboflow). Script `evaluate_models.py` tự động:
- Chuyển đổi đường dẫn YAML Roboflow (`../valid/images` → `images`) qua file tạm mà **không chỉnh sửa thư mục `dataset_test/`**.
- Ánh xạ chính xác **1-1 theo tên lớp** giữa mô hình và từng tập con, đồng thời lọc bỏ dự đoán thuộc các lớp không được gán nhãn trong tập con đó:

| Tập dữ liệu | Hệ lớp trong `data.yaml` | Ánh xạ từ mô hình `{0:car, 1:motorcycle, 2:person, 3:truck-bus, 4:train}` |
| :--- | :--- | :--- |
| **`roboflow_crossing`** (5 lớp) | `0:car`, `1:motorcycle`, `2:person`, `3:train`, `4:truck-bus` | `0→0`, `1→1`, `2→2`, `3→4` (`truck-bus`), `4→3` (`train`) |
| **`rcd_ground`** (3 lớp) | `0:car`, `1:person`, `2:truck-bus` | `0→0`, `2→1` (`person`), `3→2` (`truck-bus`) *(bỏ qua `motorcycle`, `train`)* |
| **`rcd_elevated`** (4 lớp) | `0:car`, `1:person`, `2:train`, `3:truck-bus` | `0→0`, `2→1` (`person`), `4→2` (`train`), `3→3` (`truck-bus`) *(bỏ qua `motorcycle`)* |

---

## 2. Mô tả 3 Tập Dữ liệu Kiểm thử Độc lập (`dataset_test/`)

| Tập Dữ liệu | Đường dẫn YAML | Số ảnh | Tổng Targets | Phân bố nhãn Ground Truth | Đặc điểm môi trường quan trắc |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`roboflow_crossing`** | `dataset_test/roboflow_crossing/data.yaml` | 119 | 707 | `car`: 230, `motorcycle`: 155, `person`: 222, `train`: 7, `truck-bus`: 93 | Giao cắt đường ngang dân sinh, đủ cả 5 lớp, mật độ phương tiện đông đúc. |
| **`rcd_ground`** | `dataset_test/rcd_ground/data.yaml` | 185 | 333 | `car`: 250, `person`: 59, `truck-bus`: 24 | Tuyến đường sắt mặt đất ngoại thành, góc nhìn ngang tầm mắt (3 lớp). |
| **`rcd_elevated`** | `dataset_test/rcd_elevated/data.yaml` | 100 | 198 | `car`: 133, `person`: 25, `train`: 10, `truck-bus`: 30 | Tuyến đường sắt trên cao đô thị, góc nhìn camera từ trên cao xuống (4 lớp). |
| **TỔNG CỘNG** | — | **404** | **1.238** | `car`: 613, `motorcycle`: 155, `person`: 306, `train`: 17, `truck-bus`: 147 | Bao quát đầy đủ cả 5 phân lớp thực địa đường sắt. |

---

## 3. Bảng Kết quả Đo đạc Thực tế

### 3.1. Bảng Chỉ số Trung bình Toàn diện trên 3 Tập `dataset_test` (Macro Average)
*(Trích xuất từ `results/comparison_average.csv`, chạy đồng nhất ở `batch=1`, `imgsz=640`, `device=cpu`)*

| Mô hình | Precision (%) | Recall (%) | F1-Score (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Suy luận CPU (ms/ảnh) | Tổng độ trễ (ms/ảnh) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`YOLO11s`** *(Baseline)* | 80.60% | 64.42% | 71.30% | 70.14% | 53.21% | 89.33 ms | 91.07 ms |
| **`YOLO26s`** *(Edge Real-Time)* | 81.62% | 73.39% | 76.89% | 80.24% | 61.87% | **88.59 ms** | **90.31 ms** |
| **`YOLO26m`** *(High-Accuracy)* | **83.16%** | **82.06%** | **82.58%** | **85.96%** | **72.47%** | 257.84 ms | 259.92 ms |
| **$\Delta$ (`YOLO26s` vs `YOLO11s`)** | **+1.02%** | **+8.97%** | **+5.59%** | **+10.10%** | **+8.66%** | **-0.74 ms** | **-0.76 ms** |
| **$\Delta$ (`YOLO26m` vs `YOLO26s`)** | **+1.54%** | **+8.67%** | **+5.69%** | **+5.72%** | **+10.60%** | +169.25 ms *(2.91x)* | +169.61 ms *(2.88x)* |

---

### 3.2. Bảng Phân rã Chi tiết theo Từng Tập Dữ liệu Kiểm thử
*(Trích xuất từ `results/comparison_summary.csv`)*

| Tập Dữ liệu | Mô hình | Targets | Precision (%) | Recall (%) | F1 (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Suy luận CPU (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`roboflow_crossing`** | `YOLO11s` | 707 | 85.81% | 66.38% | 74.85% | 77.53% | 52.34% | 100.97 ms |
| *(119 ảnh, 5 lớp)* | `YOLO26s` | 707 | **87.96%** | 71.43% | 78.84% | 80.62% | 55.52% | **80.91 ms** |
| | **`YOLO26m`** | 707 | 84.92% | **79.84%** | **82.30%** | **86.67%** | **64.99%** | 220.98 ms |
| **`rcd_ground`** | `YOLO11s` | 333 | 72.06% | 68.76% | 70.37% | 72.96% | 58.49% | **84.93 ms** |
| *(185 ảnh, 3 lớp)* | `YOLO26s` | 333 | **89.31%** | 73.27% | 80.50% | 82.27% | 70.14% | 88.96 ms |
| | **`YOLO26m`** | 333 | 85.91% | **86.17%** | **86.04%** | **88.86%** | **80.71%** | 242.67 ms |
| **`rcd_elevated`** | `YOLO11s` | 198 | **83.93%** | 58.11% | 68.67% | 59.93% | 48.80% | **82.08 ms** |
| *(100 ảnh, 4 lớp)* | `YOLO26s` | 198 | 67.60% | 75.48% | 71.32% | 77.84% | 59.94% | 95.89 ms |
| | **`YOLO26m`** | 198 | 78.64% | **80.18%** | **79.40%** | **82.34%** | **71.72%** | 309.88 ms |

---

### 3.3. Bảng Phân rã Hiệu năng Trung bình theo Từng Lớp Đối tượng (5 Lớp)
*(Trích xuất từ `results/comparison_per_class_avg.csv`, tính trung bình trên các tập có `Targets > 0`)*

| Lớp Đối tượng | Targets (Số tập) | `YOLO11s` mAP50 / 50-95 | `YOLO26s` mAP50 / 50-95 | `YOLO26m` mAP50 / 50-95 | `YOLO26s` vs `11s` ($\Delta$ mAP50) | `YOLO26m` vs `26s` ($\Delta$ mAP50 / 50-95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`car`** (Ô tô con) | 613 (3/3) | 79.91% / 64.05% | 82.40% / 65.59% | **90.78% / 76.90%** | +2.49% | **+8.38% / +11.31%** |
| **`motorcycle`** (Xe máy) | 155 (1/3) | 76.13% / 47.54% | 76.51% / 49.93% | **81.09% / 54.56%** | +0.38% | **+4.58% / +4.63%** |
| **`person`** (Người đi bộ) | 306 (3/3) | 83.69% / 56.67% | 87.68% / 60.71% | **91.68% / 70.58%** | +3.99% | **+4.00% / +9.87%** |
| **`train`** (Tàu hỏa) | 17 (2/3) | 53.94% / 43.20% | 81.06% / 62.78% | **81.52% / 76.53%** | **+27.12%** | **+0.46% / +13.75%** |
| **`truck-bus`** (Xe tải/Buýt) | 147 (3/3) | 57.15% / 45.43% | 70.79% / 57.80% | **79.25% / 67.97%** | **+13.64%** | **+8.46% / +10.17%** |

---

### 3.4. Đối chiếu Khả năng Tổng quát hóa: Validation Nội bộ vs Test Độc lập (Domain Shift)
*(Val nội bộ trích từ `results.csv` của từng lần huấn luyện trên 532 ảnh validation)*

| Mô hình | Val nội bộ — mAP@0.5 | Val nội bộ — mAP@0.5:0.95 | Test độc lập — mAP@0.5 | Test độc lập — mAP@0.5:0.95 | Sụt giảm mAP@0.5 | Sụt giảm mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`YOLO11s`** | 94.61% | 78.98% | 70.14% | 53.21% | -24.47% | -25.77% |
| **`YOLO26s`** | 94.68% | 79.64% | 80.24% | 61.87% | -14.44% | -17.77% |
| **`YOLO26m`** | **96.53%** | **82.32%** | **85.96%** | **72.47%** | **-10.57%** | **-9.85%** |

---

## 4. Phân tích Khoa học & Cơ sở Ra quyết định Thích ứng Phần cứng

1. **Bước nhảy vượt bậc từ thế hệ `YOLO11` sang `YOLO26`:**
   - Cùng phân khúc kích thước `small` (~`9.4–9.9M` tham số, ~`21–22 GFLOPs`, tốc độ CPU tương đương ~`88–89 ms/ảnh`), **`YOLO26s` bỏ xa `YOLO11s` tới `+10.10%` mAP@0.5** (`80.24%` vs `70.14%`) và `+8.97%` Recall (`73.39%` vs `64.42%`), thắng trên cả **5/5 phân lớp** — đặc biệt ở lớp `train` (`+27.12%`) và `truck-bus` (`+13.64%`).
2. **`YOLO26m` đạt đỉnh độ chính xác & khả năng tổng quát hóa ngoài miền:**
   - So với `YOLO26s`, **`YOLO26m`** nâng tiếp `mAP@0.5` thêm **`+5.72%`** (đạt **85.96%**) và `mAP@0.5:0.95` thêm **`+10.60%`** (đạt **72.47%**), thắng trên **3/3 tập kiểm thử** và **5/5 phân lớp** (`person` đạt **91.68%**, `car` đạt **90.78%**, `truck-bus` đạt **79.25%**).
   - Nhờ chiến lược đóng băng 6 tầng đầu (`freeze=6`), learning rate nhỏ (`1e-5`) và kế thừa trọng số phân lớp COCO (`coco_initialization`), mức sụt giảm Domain Shift của `YOLO26m` ở `mAP@0.5:0.95` chỉ còn **`-9.85%`** (thấp hơn gần một nửa so với `-17.77%` của `YOLO26s` và `-25.77%` của `YOLO11s`).
3. **Đánh đổi Độ chính xác vs Tốc độ (Accuracy–Latency Pareto Trade-off):**
   - `YOLO26m` nặng gấp **2.17x dung lượng** (`42.03 MB` vs `19.38 MB`) và gấp **3.29x GFLOPs** (`74.99` vs `22.76`), khiến độ trễ CPU cao gấp **2.91x** (`257.84 ms` vs `88.59 ms`).
   - Kết hợp với Thực nghiệm 3 (Intel Iris Xe iGPU), `YOLO26s` đạt chuẩn Real-Time mượt mà trên máy văn phòng không GPU rời, trong khi `YOLO26m` phát huy tối đa sức mạnh khi chạy trên máy trạm có GPU rời NVIDIA CUDA.
   - **Quyết định kiến trúc (Adaptive Hardware-Aware Model Selection — v1.3.0):**
     - **Có GPU rời NVIDIA CUDA** → Tự động kích hoạt **`YOLO26m`** (độ chính xác cao nhất `85.96%` mAP@0.5).
     - **Chỉ có Intel iGPU / CPU** → Tự động kích hoạt **`YOLO26s` (OpenVINO FP16)** (đảm bảo Real-Time mượt mà không rớt frame).

---

## 5. Cấu trúc Kết quả & Hướng dẫn Tái hiện Thực nghiệm

Thư mục `results/` lưu trữ đầy đủ 5 bảng CSV, 3 biểu đồ tổng hợp (`chart_3models_accuracy.png`, `chart_3models_per_class.png`, `chart_3models_complexity.png`) và toàn bộ biểu đồ chi tiết (`BoxPR_curve`, `BoxF1_curve`, `confusion_matrix`, `val_batch*_pred`) cho cả **9 tổ hợp** `{YOLO11s, YOLO26s, YOLO26m} × {roboflow_crossing, rcd_ground, rcd_elevated}`:

```bash
# Chạy lại toàn bộ kiểm thử 3 mô hình trên dataset_test (404 ảnh)
python experiments/exp1_model_comparison/evaluate_models.py --eval --batch 1

# Xem nhanh bảng tổng kết đã lưu
python experiments/exp1_model_comparison/evaluate_models.py
```
