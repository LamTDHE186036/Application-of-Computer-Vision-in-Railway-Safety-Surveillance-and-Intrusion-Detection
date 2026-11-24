# THỰC NGHIỆM 1: ĐÁNH GIÁ & SO SÁNH 6 MÔ HÌNH NHẬN DIỆN TRÊN `dataset_test`

## 1. Mục tiêu Thực nghiệm & Danh sách 6 Mô hình So sánh

Thực nghiệm đánh giá và so sánh trực tiếp **6 cấu hình mô hình** (gồm bản `YOLO26s_Pretrained` gốc và toàn bộ **5 mô hình đã huấn luyện trong thư mục `research/finetune/`**) trên 3 tập dữ liệu kiểm thử độc lập thuộc `dataset_test/` (tổng **404 ảnh**, **1238 nhãn đối tượng** thuộc 5 lớp: `car`, `motorcycle`, `person`, `train`, `truck-bus`):

1. **`YOLO26s_Pretrained`** *(Chưa finetune — COCO 80 lớp)*: `experiments/exp1_model_comparison/weights/yolo26s.pt`
2. **`YOLO11s`** *(Baseline — Finetune toàn bộ mạng)*: `research/finetune/train_yolo11s/yolo11s.pt`
3. **`YOLO26s`** *(Đề xuất — Finetune toàn bộ mạng, `freeze=None`)*: `research/finetune/train_yolo26s/yolo26s.pt`
4. **`YOLO26s_freeze3`** *(Finetune đóng băng 3 tầng đầu `model.0..2`)*: `research/finetune/train_yolo26s_freeze3/yolo26s_freeze3.pt`
5. **`YOLO26s_freeze5`** *(Finetune đóng băng 5 tầng đầu `model.0..4`)*: `research/finetune/train_yolo26s_freeze5/yolo26s_freeze5.pt`
6. **`YOLO26s_freeze10`** *(Finetune đóng băng 10 tầng Backbone `model.0..9`)*: `research/finetune/train_yolo26s_freeze10/yolo26s_freeze10.pt`

### Cơ chế đồng bộ đường dẫn & ánh xạ lớp (`MappedDetectionValidator`)
Script `evaluate_models.py` tự động xử lý:
- Chuyển đổi đường dẫn YAML Roboflow (`../valid/images` $\rightarrow$ `images`) bằng file tạm mà **không sửa thư mục `dataset_test/`**.
- Ánh xạ chính xác ID lớp theo tên lớp cho từng mô hình và từng tập dữ liệu:
  - **5 mô hình trong `research/` (5 lớp `{0:car, 1:motorcycle, 2:person, 3:truck-bus, 4:train}`)**: khớp 1-1 theo tên lớp (`train`/`truck-bus` được tự động đổi đúng ID theo `data.yaml` từng tập; lọc bỏ lớp không có nhãn trong tập con).
  - **`YOLO26s_Pretrained` (COCO 80 lớp)**: ánh xạ `person` $\rightarrow$ `person`, `car` $\rightarrow$ `car`, `motorcycle` $\rightarrow$ `motorcycle`, `train` $\rightarrow$ `train`, gộp `truck` + `bus` $\rightarrow$ `truck-bus` (kèm class-aware NMS hậu xử lý chống trùng box), và lọc bỏ 74 lớp COCO không liên quan.

---

## 2. Mô tả 3 Tập Dữ liệu Kiểm thử (`dataset_test/`)

| Tập Dữ liệu | Đường dẫn YAML | Số ảnh | Tổng Targets | Phân bố nhãn Ground Truth | Đặc điểm môi trường quan trắc |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`roboflow_crossing`** | `dataset_test/roboflow_crossing/data.yaml` | 119 | 707 | `car`: 230, `motorcycle`: 155, `person`: 222, `train`: 7, `truck-bus`: 93 | Giao cắt đường ngang dân sinh, đủ cả 5 lớp, mật độ phương tiện đông đúc. |
| **`rcd_ground`** | `dataset_test/rcd_ground/data.yaml` | 185 | 333 | `car`: 250, `person`: 59, `truck-bus`: 24 | Tuyến đường sắt mặt đất, góc nhìn ngang tầm mắt (3 lớp). |
| **`rcd_elevated`** | `dataset_test/rcd_elevated/data.yaml` | 100 | 198 | `car`: 133, `person`: 25, `train`: 10, `truck-bus`: 30 | Tuyến đường sắt trên cao đô thị, góc nhìn camera từ trên cao xuống (4 lớp). |
| **TỔNG CỘNG** | — | **404** | **1238** | `car`: 613, `motorcycle`: 155, `person`: 306, `train`: 17, `truck-bus`: 147 | Bao quát đầy đủ cả 5 phân lớp thực địa đường sắt. |

---

## 3. Bảng Kết quả Đo đạc Thực tế (6 Mô hình trên `dataset_test`)

### 3.1. Bảng Chỉ số Trung bình Toàn diện (Macro Average Across 3 Datasets)
*(Trích xuất từ `results/comparison_average.csv`, chạy `batch=1`, `imgsz=640`, `device=cpu`)*

| Mô hình | Chiến lược Huấn luyện | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Suy luận (CPU) | Thời gian val (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`YOLO26s_Pretrained`** | Pretrained COCO 80 lớp (chưa finetune) | **85.80%** | **80.85%** | **87.39%** | **73.83%** | 88.66 ms | 15.31 s |
| **`YOLO11s`** | Finetune toàn bộ (Baseline) | 80.60% | 64.42% | 70.14% | 53.21% | **82.06 ms** | **13.85 s** |
| **`YOLO26s`** *(Đề xuất)* | Finetune toàn bộ (`freeze=None`) | 81.62% | **73.39%** | **80.24%** | 61.87% | 102.60 ms | 17.27 s |
| **`YOLO26s_freeze3`** | Finetune đóng băng 3 tầng (`model.0..2`) | 76.14% | 68.06% | 70.92% | 55.75% | 123.90 ms | 20.85 s |
| **`YOLO26s_freeze5`** | Finetune đóng băng 5 tầng (`model.0..4`) | 80.27% | 69.15% | 76.18% | 58.95% | 153.73 ms | 25.29 s |
| **`YOLO26s_freeze10`** | Finetune đóng băng 10 tầng (`model.0..9`) | **84.63%** | 69.10% | 78.52% | **62.39%** | 86.11 ms | 14.64 s |

---

### 3.2. Bảng Phân rã Chi tiết theo Từng Tập Dữ liệu Kiểm thử
*(Trích xuất từ `results/comparison_summary.csv`)*

| Tập Dữ liệu | Mô hình | Targets | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Suy luận (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`roboflow_crossing`** | `YOLO26s_Pretrained` | 707 | 83.76% | 70.83% | 79.83% | **60.38%** | 86.94 ms |
| (119 ảnh, 5 lớp) | `YOLO11s` | 707 | 85.81% | 66.38% | 77.53% | 52.34% | **78.40 ms** |
| | **`YOLO26s`** | 707 | **87.96%** | 71.43% | **80.62%** | 55.52% | 92.17 ms |
| | `YOLO26s_freeze3` | 707 | 83.59% | 66.86% | 76.41% | 54.54% | 84.15 ms |
| | `YOLO26s_freeze5` | 707 | 83.62% | 71.42% | 79.16% | 52.81% | 151.04 ms |
| | `YOLO26s_freeze10` | 707 | 81.66% | **73.64%** | 80.27% | 55.22% | 84.32 ms |
| **`rcd_ground`** | **`YOLO26s_Pretrained`** | 333 | **95.11%** | **88.54%** | **94.79%** | **86.78%** | 86.45 ms |
| (185 ảnh, 3 lớp) | `YOLO11s` | 333 | 72.06% | 68.76% | 72.96% | 58.49% | **79.93 ms** |
| | `YOLO26s` | 333 | 89.31% | 73.27% | 82.27% | 70.14% | 94.43 ms |
| | `YOLO26s_freeze3` | 333 | 86.45% | 70.23% | 76.32% | 65.82% | 135.26 ms |
| | `YOLO26s_freeze5` | 333 | 88.56% | 67.51% | 79.79% | 67.35% | 154.65 ms |
| | `YOLO26s_freeze10` | 333 | 91.25% | 74.57% | 87.37% | 77.71% | 87.12 ms |
| **`rcd_elevated`** | **`YOLO26s_Pretrained`** | 198 | 78.54% | **83.17%** | **87.56%** | **74.32%** | 92.59 ms |
| (100 ảnh, 4 lớp) | `YOLO11s` | 198 | **83.93%** | 58.11% | 59.93% | 48.80% | 87.86 ms |
| | **`YOLO26s`** | 198 | 67.60% | **75.48%** | **77.84%** | **59.94%** | 121.21 ms |
| | `YOLO26s_freeze3` | 198 | 58.38% | 67.08% | 60.02% | 46.88% | 152.29 ms |
| | `YOLO26s_freeze5` | 198 | 68.63% | 68.53% | 69.60% | 56.68% | 155.49 ms |
| | `YOLO26s_freeze10` | 198 | 80.98% | 59.10% | 67.93% | 54.24% | **86.88 ms** |

---

### 3.3. Bảng Phân rã mAP@0.5 Trung bình theo Từng Lớp Đối tượng (Đủ 5 Lớp)
*(Trích xuất từ `results/comparison_per_class.csv`, trung bình trên các tập có nhãn > 0)*

| Lớp Đối tượng | Targets | `YOLO26s_Pretrained` | `YOLO11s` | `YOLO26s` | `YOLO26s_freeze3` | `YOLO26s_freeze5` | `YOLO26s_freeze10` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`car`** | 613 *(3 tập)* | **88.31%** | 79.91% | 82.40% | 81.77% | 79.98% | **83.39%** |
| **`motorcycle`** | 155 *(1 tập)* | 67.11% | 76.13% | 76.51% | 71.35% | 76.49% | **82.30%** |
| **`person`** | 306 *(3 tập)* | 86.99% | 83.69% | 87.68% | 82.11% | 85.00% | **88.89%** |
| **`train`** | 17 *(2 tập)* | **93.42%** | 53.94% | **81.06%** | 60.88% | 74.10% | 69.75% |
| **`truck-bus`** | 147 *(3 tập)* | **84.64%** | 57.15% | **70.79%** | 55.43% | 64.64% | 65.50% |

---

## 4. Phân tích Khoa học & Đánh giá Kết quả 6 Mô hình

1. **Tất cả 4 cấu hình YOLO26s trong `research/` đều vượt trội `YOLO11s`:**
   - Từ `YOLO26s_freeze3` (70.92%), `YOLO26s_freeze5` (76.18%), `YOLO26s_freeze10` (78.52%) đến **`YOLO26s` (80.24%)** đều cao hơn `YOLO11s` (70.14%), khẳng định ưu thế rõ rệt của kiến trúc YOLO26s.
2. **So sánh quy luật đóng băng tầng (`freeze=None` vs `freeze=3` vs `freeze=5` vs `freeze=10`):**
   - **`YOLO26s` (`freeze=None` — Finetune toàn bộ mạng) đạt hiệu năng tổng thể cao nhất trong các mô hình finetune**: **mAP@0.5 = 80.24%**, **Recall = 73.39%**, dẫn đầu trên cả `roboflow_crossing` (**80.62%**) lẫn `rcd_elevated` (**77.84%**) và hai lớp quan trọng `train` (**81.06%**) & `truck-bus` (**70.79%**).
   - Trong nhóm có đóng băng (`freeze=3`, `freeze=5`, `freeze=10`):
     - **`freeze=10` (78.52%)** tốt hơn `freeze=5` (76.18%) và `freeze=3` (70.92%) trên mặt bằng chung vì `freeze=10` khóa trọn vẹn tại ranh giới khối SPPF cuối Backbone (`model.9`), giúp giữ nguyên không gian đặc trưng COCO hoàn chỉnh cho Neck sử dụng (đạt Precision **84.63%** và mAP@0.5:0.95 **62.39%** cao nhất nhóm finetune). Ngược lại, khóa nửa chừng giữa Backbone (`freeze=3`, `freeze=5`) với learning rate mặc định tạo ra độ lệch biểu diễn giữa các tầng đầu cố định và các tầng giữa Backbone biến thiên mạnh.
     - Tuy nhiên, đúng như lý thuyết kiến trúc, **`freeze=5` mở khóa nhánh P4/P5 nên bắt lớp `train` tốt hơn `freeze=10`** (`train` mAP@0.5 đạt **74.10%** vs **69.75%**, Recall **75.00%** vs **50.00%**) và đạt điểm trên góc nhìn trên cao `rcd_elevated` cao hơn `freeze=10` (**69.60%** vs **67.93%**).

---

## 5. Hướng dẫn Tái hiện Thực nghiệm

```bash
# 1. Chạy lại toàn bộ đánh giá 6 mô hình trên dataset_test (404 ảnh)
python experiments/exp1_model_comparison/evaluate_models.py --eval --batch 1

# 2. Chỉ chạy đánh giá một hoặc vài mô hình chỉ định và gộp vào bảng chung
python experiments/exp1_model_comparison/evaluate_models.py --only YOLO26s_freeze3,YOLO26s_freeze5 --batch 1

# 3. Xem nhanh bảng tổng kết đã lưu sẵn
python experiments/exp1_model_comparison/evaluate_models.py
```
