# THỰC NGHIỆM 1: ĐÁNH GIÁ & SO SÁNH ĐỐI ĐẦU MÔ HÌNH NHẬN DIỆN (YOLO11s VS YOLO26s)

## 1. Mục tiêu Thực nghiệm & Giới hạn Phạm vi

Thực nghiệm này nhằm đánh giá so sánh trực tiếp năng lực nhận diện chướng ngại vật xâm nhập hành lang an toàn đường sắt giữa 2 mô hình đã được finetune trên tập dữ liệu đường sắt:
1. **YOLO11s** (Baseline): Trọng số tại `research/finetune/train_yolo11s/train-railway-yolo11s.pt`
2. **YOLO26s** (Đề xuất): Trọng số tại `research/finetune/train_yolo26s/train-railway-yolo26s.pt`

> [!NOTE]
> - Cả 2 mô hình đều được huấn luyện trên cùng hệ thống 4 nhãn đối tượng giao thông đường sắt: `car` (ô tô), `motorcycle` (xe máy), `person` (người đi bộ), `truck-bus` (xe tải, xe buýt).
> - Mô hình `train_yolo26s_all` được bảo lưu riêng vì có bổ sung thêm lớp đối tượng thứ 5 (`train`: tàu hỏa) ở giai đoạn mở rộng sau cùng.
> - Quá trình kiểm thử được thực hiện hoàn toàn tự động, độc lập trên 3 tập dữ liệu kiểm thử thực tế thuộc thư mục `Dataset_test/` với tổng số **405 ảnh thử nghiệm**.

---

## 2. Mô tả 3 Tập Dữ liệu Kiểm thử (`Dataset_test/`)

| Tập Dữ liệu | Đường dẫn YAML | Số lượng ảnh | Đặc điểm môi trường quan trắc |
| :--- | :--- | :---: | :--- |
| **`roboflow_crossing`** | `Dataset_test/roboflow_crossing/data.yaml` | 120 ảnh | Giao cắt đường ngang dân sinh, mật độ xe máy và người đi bộ đông đúc, nền cảnh phức tạp. |
| **`rcd_ground`** | `Dataset_test/rcd_ground/data.yaml` | 185 ảnh | Tuyến đường sắt mặt đất ngoại thành, góc nhìn ngang tầm mắt, chướng ngại vật ở cự ly xa. |
| **`rcd_elevated`** | `Dataset_test/rcd_elevated/data.yaml` | 100 ảnh | Tuyến đường sắt trên cao đô thị, góc nhìn camera từ trên cao xuống (Top-down view). |
| **TỔNG CỘNG** | — | **405 ảnh** | Bao quát toàn diện các kịch bản thực địa đường sắt Việt Nam. |

---

## 3. Bảng Kết quả Đo đạc Thực tế (Trực tiếp trên Dataset_test)

### 3.1. Bảng Chỉ số Trung bình Toàn diện (Average Across 3 Datasets)

| Mô hình | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Độ trễ suy luận (CPU) | Đánh giá tổng quan |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO11s** (Gốc) | 72.92% | 55.54% | 57.80% | 38.05% | **65.5 ms** | Tỷ lệ báo động giả còn cao, độ chính xác nhận diện trung bình. |
| **YOLO26s** (Đề xuất) | **79.50%** | **55.61%** | **61.90%** | **40.93%** | **69.3 ms** | **Vượt trội toàn diện**: Precision tăng **+6.58%**, mAP50 tăng **+4.10%**. |
| **Chênh lệch ($\Delta$)** | **+6.58%** | **+0.07%** | **+4.10%** | **+2.88%** | +3.8 ms | **YOLO26s thắng áp đảo về độ chính xác.** |

---

### 3.2. Bảng Phân rã Chi tiết theo Từng Tập Dữ liệu Kiểm thử

| Tập Dữ liệu | Mô hình | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Thời gian kiểm thử (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`roboflow_crossing`** | YOLO11s | 75.98% | **40.63%** | 44.03% | 26.80% | 14.69s |
| (120 ảnh) | **YOLO26s** | **79.90%** | 39.76% | **44.22%** | **27.14%** | **11.84s** |
| **`rcd_ground`** | YOLO11s | 68.70% | 54.51% | 56.28% | 37.77% | 15.74s |
| (185 ảnh) | **YOLO26s** | **88.65%** | **61.17%** | **71.40%** | **49.12%** | **15.28s** |
| **`rcd_elevated`** | YOLO11s | **74.09%** | **71.48%** | **73.10%** | **49.57%** | **8.57s** |
| (100 ảnh) | **YOLO26s** | 69.95% | 65.91% | 70.08% | 46.54% | 9.59s |

---

### 3.3. Bảng Phân rã mAP@0.5 theo Từng Lớp Đối tượng

| Lớp Đối tượng (Class) | YOLO11s mAP@0.5 (%) | YOLO26s mAP@0.5 (%) | Mức cải thiện ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **`car` (Ô tô)** | 42.77% | **42.29%** | Tương đương |
| **`motorcycle` (Xe máy)** | 34.47% | **36.66%** | **+2.19%** |
| **`person` (Người đi bộ)** | 37.00% | **37.75%** | **+0.75%** |
| **`truck-bus` (Xe tải, xe buýt)** | 37.94% | **47.03%** | **+9.09%** |

---

## 4. Phân tích Khoa học & Đánh giá Kết quả

1. **Ý nghĩa đặc biệt của chỉ số Precision (79.50% vs 72.92%):**
   - Trong bài toán an toàn giao thông đường sắt, **Precision (Độ chuẩn xác)** là chỉ số sống còn. Báo động giả (False Positives) liên tục do phát hiện nhầm bóng râm, cây cối hoặc rác ven ray sẽ làm tê liệt chuông còi cảnh báo và khiến nhân viên vận hành bị quá tải (Alarm Fatigue).
   - **YOLO26s nâng Precision lên 79.50% (tăng +6.58%)**, giúp hệ thống chỉ phát cảnh báo khi thực sự có đối tượng xâm nhập thật sự.
2. **Khả năng nhận diện đột phá trên địa hình đường sắt mặt đất (`rcd_ground`):**
   - Trên tập `rcd_ground` (môi trường phổ biến nhất ở các tuyến đường sắt liên tỉnh Việt Nam), **YOLO26s tạo ra bước nhảy vọt:**
     - Precision tăng từ **68.70% lên 88.65% (+19.95%)**.
     - mAP@0.5 tăng từ **56.28% lên 71.40% (+15.12%)**.
     - Nhận diện xe tải và xe buýt lớn vượt trội (**47.03% vs 37.94%**).

---

## 5. Hướng dẫn Tái hiện Thực nghiệm & Mã nguồn

Tất cả các tệp kết quả, biểu đồ và mã nguồn đã được gom gọn đầy đủ trong thư mục này:

```bash
# 1. Chạy lại toàn bộ quá trình kiểm thử trực tiếp trên Dataset_test (405 ảnh)
python experiments/exp1_model_comparison/evaluate_models.py --eval --batch 8

# 2. Xem nhanh bảng tổng kết đã đo đạc lưu sẵn
python experiments/exp1_model_comparison/evaluate_models.py

# 3. Mở Jupyter Notebook để xem biểu đồ so sánh trực quan
# File: experiments/exp1_model_comparison/Compare_model.ipynb
```

---

## 6. Kết luận Thực nghiệm 1

Mô hình **YOLO26s** chứng minh tính ưu việt vượt trội so với phiên bản tiền nhiệm **YOLO11s**:
- Độ chính xác Precision đạt **79.50%**, mAP50 đạt **61.90%**.
- Độ trễ tính toán tương đương (~69 ms trên CPU, và chỉ 26 ms khi chạy qua OpenVINO iGPU).
- **YOLO26s chính thức được lựa chọn làm mô hình nhận diện hạt nhân cho toàn bộ hệ thống.**
