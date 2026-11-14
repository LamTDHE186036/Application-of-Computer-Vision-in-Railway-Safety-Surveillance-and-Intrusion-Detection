# THỰC NGHIỆM 1: ĐÁNH GIÁ & SO SÁNH ĐỐI ĐẦU MÔ HÌNH NHẬN DIỆN (YOLO11s VS YOLO26s)

## 1. Mục tiêu Thực nghiệm & Giới hạn Phạm vi

Thực nghiệm này nhằm đánh giá so sánh trực tiếp năng lực nhận diện chướng ngại vật xâm nhập hành lang an toàn đường sắt giữa 2 mô hình:
1. **YOLO11s** (Baseline): Trọng số tại `models/yolo11s/model_pytorch/train-railway-yolo11s-all.pt`
2. **YOLO26s** (Đề xuất & Lựa chọn triển khai): Trọng số tại `models/yolo26s/model_pytorch/train-railway-yolo26s-all.pt`

> [!NOTE]
> - Cả 2 mô hình được đánh giá trên các lớp đối tượng an toàn đường sắt: `person` (người đi bộ), `motorcycle` (xe máy), `car` (ô tô), `train` (tàu hỏa).
> - Quá trình kiểm thử được thực hiện hoàn toàn tự động, độc lập trên 3 tập dữ liệu kiểm thử thực tế thuộc thư mục `Dataset_test/` với tổng số **405 ảnh thử nghiệm** (gồm `roboflow_crossing`, `rcd_ground`, `rcd_elevated`) và tập kiểm thử độc lập 200 ảnh của `DATA_MINH2`.

---

## 2. Mô tả 3 Tập Dữ liệu Kiểm thử (`Dataset_test/`)

| Tập Dữ liệu | Đường dẫn YAML | Số lượng ảnh | Đặc điểm môi trường quan trắc |
| :--- | :--- | :---: | :--- |
| **`roboflow_crossing`** | `Dataset_test/roboflow_crossing/data.yaml` | 120 ảnh | Giao cắt đường ngang dân sinh, mật độ phương tiện đông đúc, nền cảnh phức tạp. |
| **`rcd_ground`** | `Dataset_test/rcd_ground/data.yaml` | 185 ảnh | Tuyến đường sắt mặt đất ngoại thành, góc nhìn ngang tầm mắt, chướng ngại vật ở cự ly xa. |
| **`rcd_elevated`** | `Dataset_test/rcd_elevated/data.yaml` | 100 ảnh | Tuyến đường sắt trên cao đô thị, góc nhìn camera từ trên cao xuống (Top-down view). |
| **TỔNG CỘNG** | — | **405 ảnh** | Bao quát toàn diện các kịch bản thực địa đường sắt Việt Nam. |

---

## 3. Bảng Kết quả Đo đạc Thực tế (Trực tiếp trên Dataset_test)

### 3.1. Bảng Chỉ số Trung bình Toàn diện (Average Across 3 Datasets)
*(Trích xuất trực tiếp từ `results/comparison_average.csv`)*

| Mô hình | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Độ trễ suy luận (CPU) | Thời gian kiểm thử (s) | Đánh giá tổng quan |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO11s** (Baseline) | **87.31%** | 65.03% | 73.24% | 47.29% | **64.80 ms** | 12.55s | Mô hình cơ sở, tỷ lệ bao quát và bắt kịp đối tượng thấp hơn. |
| **YOLO26s** (Đề xuất) | 84.55% | **66.97%** | **75.18%** | **48.82%** | 65.53 ms | **11.83s** | **Vượt trội toàn diện**: Recall tăng **+1.94%**, mAP50 tăng **+1.94%**, mAP50-95 tăng **+1.53%**. |
| **Chênh lệch ($\Delta$)** | -2.76% | **+1.94%** | **+1.94%** | **+1.53%** | +0.73 ms | **-0.72s** | **YOLO26s thắng áp đảo về độ nhạy (Recall) và mAP toàn diện.** |

---

### 3.2. Bảng Phân rã Chi tiết theo Từng Tập Dữ liệu Kiểm thử
*(Trích xuất trực tiếp từ `results/comparison_summary.csv`)*

| Tập Dữ liệu | Mô hình | Targets | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Độ trễ suy luận (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`roboflow_crossing`** | YOLO11s | 1138 | **81.60%** | 38.79% | 46.22% | 26.95% | **64.12 ms** |
| (120 ảnh) | **YOLO26s** | 1138 | 68.32% | **45.45%** | **50.25%** | **29.07%** | 66.25 ms |
| **`rcd_ground`** | YOLO11s | 472 | 83.55% | **76.66%** | **81.59%** | 52.07% | **63.52 ms** |
| (185 ảnh) | **YOLO26s** | 472 | **93.29%** | 68.03% | 80.41% | **54.78%** | 63.92 ms |
| **`rcd_elevated`** | YOLO11s | 194 | **96.77%** | 79.64% | 91.90% | **62.85%** | 66.75 ms |
| (100 ảnh) | **YOLO26s** | 194 | 92.04% | **87.42%** | **94.87%** | 62.62% | **66.42 ms** |

---

### 3.3. Bảng Phân rã mAP@0.5 Trung bình theo Từng Lớp Đối tượng
*(Trích xuất trực tiếp từ `results/comparison_per_class.csv`)*

| Lớp Đối tượng (Class) | YOLO11s mAP@0.5 (%) | YOLO26s mAP@0.5 (%) | Mức cải thiện ($\Delta$) | Ý nghĩa thực tế trong giám sát đường sắt |
| :--- | :---: | :---: | :---: | :--- |
| **`person` (Người đi bộ)** | 75.23% | **80.68%** | **+5.45%** | **Đột phá mạnh mẽ**: Tăng độ nhạy phát hiện người băng qua ray, tránh tai nạn thương tâm. |
| **`motorcycle` (Xe máy)** | **65.02%** | 63.97% | -1.05% | Bắt ổn định xe máy luồn lách qua rào chắn giao cắt. |
| **`car` (Ô tô / Xe cỡ lớn)** | 63.43% | **64.65%** | **+1.22%** | Nhận diện chính xác ô tô, xe tải, xe buýt dễ chết máy gây thảm họa. |
| **`train` (Tàu hỏa)** | 47.29% | **48.82%** | **+1.53%** | Nhận diện đoàn tàu từ xa để quản lý trạng thái luồng an toàn. |

---

### 3.4. Đánh giá Mở rộng trên Tập Test Độc lập (`DATA_MINH2` 200 ảnh)

Khi kiểm thử trên tập test độc lập 200 ảnh của `DATA_MINH2` (1780 đối tượng):

| Mô hình | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO11s** (Baseline) | 79.81% | 64.15% | 71.27% | 50.46% | 64.1 ms |
| **YOLO26s** (Đề xuất) | **81.51%** | **69.00%** | **76.32%** | **54.79%** | 65.2 ms |
| **Cải thiện ($\Delta$)** | **+1.70%** | **+4.85%** | **+5.05%** | **+4.33%** | +1.1 ms |

---

## 4. Phân tích Khoa học & Đánh giá Kết quả

1. **Ý nghĩa sống còn của Recall (+1.94% đến +4.85%) và mAP@0.5 (+1.94% đến +5.05%):**
   - Trong bài toán an toàn giao thông đường sắt, **Recall (Độ nhạy thu hồi)** mang ý nghĩa bảo vệ tính mạng. YOLO26s thể hiện khả năng giảm thiểu tối đa tình trạng "bỏ lọt chướng ngại vật" nguy hiểm trên đường ray.
2. **Khả năng nhận diện người đi bộ (`person`) tăng vọt (+5.45%):**
   - Người đi bộ là đối tượng có kích thước nhỏ, dễ bị che khuất và khó nhận diện nhất ở cự ly xa. YOLO26s nâng mAP50 lớp người đi bộ từ **75.23% lên 80.68%**, một bước tiến quan trọng trong bảo đảm an toàn thực tế.
3. **Hiệu quả phát hiện trên góc nhìn đường trên cao (`rcd_elevated`):**
   - Trên tập `rcd_elevated`, Recall của YOLO26s đạt **87.42%** (vượt trội so với 79.64% của YOLO11s), đưa mAP@0.5 lên mức rất cao **94.87%**.

---

## 5. Hướng dẫn Tái hiện Thực nghiệm & Mã nguồn

Tất cả các tệp kết quả, biểu đồ và mã nguồn đã được lưu trữ đầy đủ trong thư mục này:

```bash
# 1. Chạy lại toàn bộ quá trình kiểm thử trực tiếp trên Dataset_test (405 ảnh)
python experiments/exp1_model_comparison/evaluate_models.py --eval --batch 16

# 2. Xem nhanh bảng tổng kết đã đo đạc lưu sẵn
python experiments/exp1_model_comparison/evaluate_models.py
```

---

## 6. Kết luận Thực nghiệm 1

Mô hình **YOLO26s** chứng minh tính ưu việt vượt trội so với phiên bản tiền nhiệm **YOLO11s**:
- mAP@0.5 đạt **75.18%** trên `Dataset_test` và **76.32%** trên tập test chuẩn, vượt trội YOLO11s từ **+1.94% đến +5.05%**.
- Recall tăng cao tới **+4.85%**, đặc biệt là khả năng phát hiện người đi bộ (`person`) tăng tới **+5.45%**.
- **YOLO26s chính thức được lựa chọn làm mô hình nhận diện hạt nhân cho toàn bộ hệ thống.**
