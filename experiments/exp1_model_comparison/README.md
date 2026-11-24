# THỰC NGHIỆM 1: ĐÁNH GIÁ & SO SÁNH ĐỐI ĐẦU MÔ HÌNH NHẬN DIỆN (YOLO11s VS YOLO26s)

## 1. Mục tiêu Thực nghiệm & Cấu hình Đánh giá

Thực nghiệm nhằm đánh giá so sánh trực tiếp năng lực nhận diện chướng ngại vật xâm nhập hành lang an toàn đường sắt giữa **2 kiến trúc mô hình** được huấn luyện trên cùng bộ dữ liệu `Dataset_all` (`2,125` ảnh train, `532` ảnh val, `50` epochs, `imgsz=640`, `batch=16`, `seed=42`) với hệ **5 lớp đối tượng** (`0: car`, `1: motorcycle`, `2: person`, `3: truck-bus`, `4: train` — gồm 4 lớp chướng ngại vật xâm nhập và 1 lớp tàu hỏa):

1. **YOLO11s (Baseline):** `research/finetune/train_yolo11s/yolo11s.pt` *(100 layers fused, 9.41M params, 21.4 GFLOPs)*
2. **YOLO26s (Đề xuất & Triển khai):** `models/yolo26s/model_pytorch/train-railway-yolo26s-all.pt` *(120 layers fused, 9.47M params, 20.8 GFLOPs — trùng với `research/finetune/train_yolo26s/yolo26s.pt`)*

### Cơ chế đồng bộ đường dẫn & ánh xạ lớp (`MappedDetectionValidator`)
Bộ kiểm thử độc lập `dataset_test/` đã chuẩn hóa đầy đủ 5 lớp tương thích với mô hình (`car`, `motorcycle`, `person`, `train`, `truck-bus`), nhưng thứ tự ID và tập lớp có mặt ở mỗi tập con khác nhau (do xuất từ Roboflow). Script `evaluate_models.py` tự động xử lý:
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
| **TỔNG CỘNG** | — | **404** | **1238** | `car`: 613, `motorcycle`: 155, `person`: 306, `train`: 17, `truck-bus`: 147 | Bao quát đầy đủ cả 5 phân lớp thực địa đường sắt. |

---

## 3. Bảng Kết quả Đo đạc Thực tế

### 3.1. Bảng Chỉ số Trung bình Toàn diện trên 3 Tập `dataset_test` (Macro Average)
*(Trích xuất từ `results/comparison_average.csv`, chạy ở `batch=1`, `imgsz=640`, `device=cpu`)*

| Mô hình | Precision (%) | Recall (%) | F1-Score (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Suy luận CPU (ms/ảnh) | Thời gian val (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11s** (Baseline) | 80.60% | 64.42% | 71.61% | 70.14% | 53.21% | **82.06 ms** | **13.85 s** |
| **YOLO26s** (Đề xuất) | **81.62%** | **73.39%** | **77.29%** | **80.24%** | **61.87%** | 102.60 ms | 17.27 s |
| **Chênh lệch (Δ)** | **+1.02%** | **+8.97%** | **+5.68%** | **+10.10%** | **+8.66%** | +20.54 ms | +3.42 s |

---

### 3.2. Bảng Phân rã Chi tiết theo Từng Tập Dữ liệu Kiểm thử
*(Trích xuất từ `results/comparison_summary.csv`)*

| Tập Dữ liệu | Mô hình | Targets | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Suy luận (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`roboflow_crossing`** | YOLO11s | 707 | 85.81% | 66.38% | 77.53% | 52.34% | **78.40 ms** |
| *(119 ảnh, 5 lớp)* | **YOLO26s** | 707 | **87.96%** | **71.43%** | **80.62%** | **55.52%** | 92.17 ms |
| **`rcd_ground`** | YOLO11s | 333 | 72.06% | 68.76% | 72.96% | 58.49% | **79.93 ms** |
| *(185 ảnh, 3 lớp)* | **YOLO26s** | 333 | **89.31%** | **73.27%** | **82.27%** | **70.14%** | 94.43 ms |
| **`rcd_elevated`** | YOLO11s | 198 | **83.93%** | 58.11% | 59.93% | 48.80% | **87.86 ms** |
| *(100 ảnh, 4 lớp)* | **YOLO26s** | 198 | 67.60% | **75.48%** | **77.84%** | **59.94%** | 121.21 ms |

---

### 3.3. Bảng Phân rã Hiệu năng Trung bình theo Từng Lớp Đối tượng (5 Lớp)
*(Trích xuất từ `results/comparison_per_class.csv`, trung bình trên các tập có nhãn > 0)*

| Lớp Đối tượng | Targets (Số tập) | YOLO11s P / R (%) | YOLO26s P / R (%) | YOLO11s mAP@0.5 (%) | YOLO26s mAP@0.5 (%) | Δ mAP@0.5 | YOLO11s mAP50-95 (%) | YOLO26s mAP50-95 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`car`** (Ô tô con) | 613 (3/3 tập) | 92.22 / **68.48** | **92.39** / 67.04 | 79.91% | **82.40%** | **+2.49%** | 64.05% | **65.59%** |
| **`motorcycle`** (Xe máy) | 155 (1/3 tập) | 75.73 / 61.29 | **88.15** / **62.58** | 76.13% | **76.51%** | **+0.38%** | 47.54% | **49.93%** |
| **`person`** (Người đi bộ) | 306 (3/3 tập) | **86.03** / 74.00 | 81.63 / **82.36** | 83.69% | **87.68%** | **+3.99%** | 56.67% | **60.71%** |
| **`train`** (Tàu hỏa) | 17 (2/3 tập) | **95.25** / 50.00 | 77.36 / **78.88** | 53.94% | **81.06%** | **+27.12%** | 43.20% | **62.78%** |
| **`truck-bus`** (Xe tải/Buýt) | 147 (3/3 tập) | 59.99 / 60.63 | **71.05** / **70.12** | 57.15% | **70.79%** | **+13.64%** | 45.43% | **57.80%** |

---

### 3.4. Đối chiếu Khả năng Tổng quát hóa: Validation Nội bộ vs Test Độc lập (`dataset_test`)
*(Số liệu Validation nội bộ trích từ `research/finetune/train_yolo11s/.../results.csv` và `train_yolo26s/.../results.csv` trên 532 ảnh / 4,586 đối tượng)*

| Mô hình | Val nội bộ — mAP@0.5 | Val nội bộ — mAP@0.5:0.95 | Test độc lập — mAP@0.5 | Test độc lập — mAP@0.5:0.95 | Độ sụt giảm mAP@0.5 (Domain Shift) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO11s** | 94.61% | 78.98% | 70.14% | 53.21% | -24.47% *(Tổng quát hóa yếu hơn)* |
| **YOLO26s** | **94.68%** | **79.64%** | **80.24%** | **61.87%** | **-14.44%** *(Tổng quát hóa vượt trội +10.03%)* |

---

## 4. Phân tích Khoa học & Đánh giá Kết quả

1. **Khả năng tổng quát hóa vượt trội trên dữ liệu thực tế ngoài miền huấn luyện:**
   - Trên tập Validation nội bộ (`532` ảnh), hai mô hình đạt độ chính xác gần tương đương (`~94.6% – 94.7%` mAP@0.5). Tuy nhiên, khi chuyển sang 3 tập kiểm thử độc lập `dataset_test` với điều kiện ánh sáng và góc camera hoàn toàn mới, **YOLO26s giữ vững mAP@0.5 đạt 80.24%** trong khi YOLO11s giảm xuống **70.14%** (chênh lệch **+10.10% mAP@0.5** và **+8.66% mAP@0.5:0.95**).
2. **YOLO26s thắng toàn diện 5/5 lớp và đặc biệt bứt phá ở 2 lớp khó (`train` & `truck-bus`):**
   - **Lớp `train` (+27.12% mAP@0.5):** Trên tập đường sắt trên cao `rcd_elevated`, YOLO11s gần như bỏ sót đoàn tàu (`Recall = 0%`, `mAP@0.5 = 8.39%`), trong khi YOLO26s nhận diện ổn định (`mAP@0.5 = 62.63%`, và đạt `99.50%` trên `roboflow_crossing`). Đây là điều kiện sống còn cho chế độ cảnh báo *Khi tàu tới* (`train_approaching`).
   - **Lớp `truck-bus` (+13.64% mAP@0.5):** Phương tiện cỡ lớn là nguồn rủi ro tai nạn đường sắt nghiêm trọng nhất; YOLO26s nâng cả Precision (`59.99%` → `71.05%`) lẫn Recall (`60.63%` → `70.12%`).
   - **Lớp `person` (+3.99% mAP@0.5):** Đạt mAP@0.5 cao nhất trong 5 lớp (**87.68%**) với Recall trung bình **82.36%** (đạt **100%** Recall trên `rcd_elevated`).
3. **Độ ổn định trên mọi góc lắp đặt camera:**
   - YOLO26s duy trì mAP@0.5 đồng đều trên cả 3 kịch bản từ **77.84% đến 82.27%** (biên độ chỉ `4.43%`), không bị sụt giảm mạnh ở góc nhìn trên cao như YOLO11s (`59.93% – 77.53%`). Về tốc độ suy luận trên phần cứng biên, sự chênh lệch trên CPU thuần sẽ được triệt tiêu hoàn toàn sau khi tối ưu bằng **Intel OpenVINO FP16 trên Intel Iris Xe iGPU** (Thực nghiệm 3 — đạt **35.5 FPS / 28.16 ms**).

---

## 5. Cấu trúc Kết quả & Hướng dẫn Tái hiện Thực nghiệm

Thư mục `results/` lưu trữ đầy đủ số liệu CSV và toàn bộ biểu đồ đánh giá (đường cong `BoxPR_curve`, `BoxF1_curve`, `BoxP_curve`, `BoxR_curve`, `confusion_matrix`, ảnh đối chiếu `val_batch*_labels/pred`) cho cả 6 tổ hợp `{YOLO11s, YOLO26s} × {roboflow_crossing, rcd_ground, rcd_elevated}`:

```bash
# 1. Chạy lại toàn bộ quá trình kiểm thử YOLO11s vs YOLO26s trên dataset_test (404 ảnh)
python experiments/exp1_model_comparison/evaluate_models.py --eval --batch 1

# 2. Chạy nhanh hơn với batch=16
python experiments/exp1_model_comparison/evaluate_models.py --eval --batch 16

# 3. Xem nhanh bảng tổng kết đã lưu sẵn
python experiments/exp1_model_comparison/evaluate_models.py
```

---

## 6. Kết luận Thực nghiệm 1

Mô hình **YOLO26s** chứng minh tính ưu việt áp đảo so với **YOLO11s** trên cả 3 tập dữ liệu kiểm thử độc lập `dataset_test` (5 lớp):
- **mAP@0.5** đạt **80.24%** (**+10.10%**), **mAP@0.5:0.95** đạt **61.87%** (**+8.66%**), **Recall** đạt **73.39%** (**+8.97%**), **F1-Score** đạt **77.29%** (**+5.68%**), **Precision** đạt **81.62%** (**+1.02%**).
- Thắng tuyệt đối trên **cả 5/5 phân lớp** (`car`, `motorcycle`, `person`, `train`, `truck-bus`) và vượt trội rõ rệt về khả năng tổng quát hóa ngoài miền huấn luyện.
- **YOLO26s chính thức được lựa chọn làm mô hình nhận diện hạt nhân cho toàn bộ hệ thống.**
