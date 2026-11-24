# BÁO CÁO NGHIÊN CỨU THỰC NGHIỆM VÀ ĐÁNH GIÁ HỆ THỐNG
## Ứng dụng Thị giác Máy tính trong Giám sát An toàn Đường sắt và Phát hiện Xâm nhập

Tài liệu này tổng hợp toàn bộ phương pháp luận, quy trình triển khai và số liệu thực nghiệm đo đạc thực tế của 03 bài toán nghiên cứu cốt lõi trong hệ thống. Cấu trúc được chuẩn hóa để có thể trích dẫn và tích hợp trực tiếp vào **Chương 4 (Nghiên cứu Thực nghiệm và Đánh giá Kết quả)** của báo cáo / khóa luận tốt nghiệp.

---

## MỤC LỤC BÁO CÁO THỰC NGHIỆM

1. [Cấu trúc Thư mục Thực nghiệm](#1-cấu-trúc-thư-mục-thực-nghiệm)
2. [Thực nghiệm 1: Đánh giá & So sánh Kiến trúc Mô hình Nhận diện (YOLO11s vs YOLO26s)](#2-thực-nghiệm-1-đánh-giá--so-sánh-kiến-trúc-mô-hình-nhận-diện-yolo11s-vs-yolo26s)
3. [Thực nghiệm 2: Đánh giá & So sánh Giải thuật Tracking Đa Đối tượng (MOT)](#3-thực-nghiệm-2-đánh-giá--so-sánh-giải-thuật-tracking-đa-đối-tượng-mot)
4. [Thực nghiệm 3: Đánh giá Tăng tốc Phần cứng & Tối ưu hóa Suy luận (PyTorch vs Intel OpenVINO)](#4-thực-nghiệm-3-đánh-giá-tăng-tốc-phần-cứng--tối-ưu-hóa-suy-luận-pytorch-vs-intel-openvino)
5. [Tổng kết Kiến trúc Công nghệ Tối ưu của Hệ thống](#5-tổng-kết-kiến-trúc-công-nghệ-tối-ưu-của-hệ-thống)

---

## 1. Cấu trúc Thư mục Thực nghiệm

Thư mục `experiments/` được tổ chức khoa học, phân tách độc lập thành 3 module tương ứng với 3 bài toán:

```
experiments/
├── README.md                           # Báo cáo tổng hợp toàn bộ 3 thực nghiệm
│
├── exp1_model_comparison/              # THỰC NGHIỆM 1: SO SÁNH MÔ HÌNH NHẬN DIỆN (YOLO11s vs YOLO26s)
│   ├── README.md                       # Báo cáo chi tiết Thực nghiệm 1
│   ├── evaluate_models.py              # Script nạp số liệu & chạy kiểm thử trực tiếp trên Dataset_test
│   └── results/                        # Toàn bộ file CSV số liệu và biểu đồ đường cong PR/F1/Confusion Matrix
│
├── exp2_tracker_comparison/            # THỰC NGHIỆM 2: SO SÁNH GIẢI THUẬT TRACKING (Dùng YOLO26s)
│   ├── README.md                       # Báo cáo chi tiết Thực nghiệm 2
│   ├── benchmark_tracker.py            # Script đo đạc FPS, Latency, Overhead, Track stability
│   ├── bytetrack.yaml                  # File cấu hình tham số liên kết của ByteTrack
│   └── results/                        # Bảng kết quả CSV so sánh các giải thuật tracking
│
└── exp3_hardware_acceleration/         # THỰC NGHIỆM 3: TĂNG TỐC SUY LUẬN PHẦN CỨNG (OpenVINO iGPU)
    ├── README.md                       # Báo cáo chi tiết Thực nghiệm 3
    ├── benchmark_hardware.py           # Script đo đạc PyTorch CPU vs OpenVINO CPU & iGPU
    └── results/                        # Bảng kết quả CSV đo đạc tăng tốc phần cứng
```

---

## 2. Thực nghiệm 1: Đánh giá & So sánh Kiến trúc Mô hình Nhận diện (YOLO11s vs YOLO26s)

### 2.1. Mục tiêu & Dữ liệu Kiểm thử
Thực nghiệm nhằm so sánh trực tiếp hiệu năng nhận diện chướng ngại vật xâm nhập giữa 2 kiến trúc được huấn luyện trên dữ liệu an toàn đường sắt đầy đủ 5 lớp (`0: car`, `1: motorcycle`, `2: person`, `3: truck-bus`, `4: train`):
- **YOLO11s** *(Baseline)*: `research/finetune/train_yolo11s/yolo11s.pt`
- **YOLO26s** *(Đề xuất)*: `models/yolo26s/model_pytorch/train-railway-yolo26s-all.pt` *(trùng với `research/finetune/train_yolo26s/yolo26s.pt`)*

Hai mô hình được đánh giá chéo độc lập trên 3 tập dữ liệu kiểm thử thực tế thuộc thư mục `dataset_test/` đã chuẩn hóa đầy đủ 5 lớp tương thích (tổng **404 ảnh**, **1238 nhãn đối tượng**):
1. `roboflow_crossing`: Giao cắt đường ngang dân sinh, đủ cả 5 lớp (119 ảnh, 707 đối tượng).
2. `rcd_ground`: Tuyến đường sắt mặt đất, 3 lớp `car`, `person`, `truck-bus` (185 ảnh, 333 đối tượng).
3. `rcd_elevated`: Tuyến đường sắt trên cao đô thị, 4 lớp `car`, `person`, `train`, `truck-bus` (100 ảnh, 198 đối tượng).

> **Cơ chế đồng bộ lớp trong `evaluate_models.py` (`MappedDetectionValidator`):** Tự động khớp chính xác **1-1 theo tên lớp** cho từng tập (xử lý thứ tự ID đảo vị trí giữa `train` và `truck-bus`, lọc bỏ lớp không gán nhãn trong tập con và xử lý đường dẫn YAML Roboflow mà không sửa `dataset_test/`).

### 2.2. Bảng Tổng hợp Kết quả Đánh giá Trung bình (Average Metrics)
*(Số liệu trích xuất từ `exp1_model_comparison/results/comparison_average.csv`, chạy `batch=1`, `imgsz=640`, `device=cpu`)*

| Mô hình | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Độ trễ suy luận (ms/ảnh) | Đánh giá tổng quan |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO11s** (Baseline) | 80.60% | 64.42% | 70.14% | 53.21% | **82.06 ms** | Mô hình cơ sở, hạn chế rõ rệt ở `truck-bus` và `train` góc nhìn trên cao. |
| **YOLO26s** (Đề xuất) | **81.62%** | **73.39%** | **80.24%** | **61.87%** | 102.60 ms | **Vượt trội toàn diện**: mAP@0.5 vượt mốc **80%** (**+10.10%**), Recall tăng **+8.97%**, mAP@0.5:0.95 tăng **+8.66%**. |
| **Chênh lệch ($\Delta$)** | **+1.02%** | **+8.97%** | **+10.10%** | **+8.66%** | +20.54 ms | **YOLO26s thắng áp đảo trên toàn bộ các chỉ số độ chính xác.** |

### 2.3. Hiệu năng theo Từng Phân lớp Đối tượng (Per-Class Average mAP@0.5 — Đủ 5 Lớp)
*(Số liệu trích xuất từ `exp1_model_comparison/results/comparison_per_class.csv`, trung bình trên các tập có nhãn > 0)*

| Lớp Đối tượng (`Class`) | Tổng Targets | YOLO11s mAP@0.5 (%) | YOLO26s mAP@0.5 (%) | Mức độ cải thiện ($\Delta$) | Ý nghĩa thực tiễn an toàn đường sắt |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Người đi bộ (`person`)** | 306 (3/3 tập) | 83.69% | **87.68%** | **+3.99%** | Bắt rất nhạy người đi bộ băng qua hoặc đi dọc ray (Recall TB **82.36%**). |
| **Ô tô con (`car`)** | 613 (3/3 tập) | 79.91% | **82.40%** | **+2.49%** | Bắt chuẩn ô tô con kẹt trên ray (Precision TB **92.39%**). |
| **Tàu hỏa (`train`)** | 17 (2/3 tập) | 53.94% | **81.06%** | **+27.12%** | **Đột phá lớn**: Nhận diện chuẩn đoàn tàu kể cả ở góc nhìn từ trên cao. |
| **Xe máy (`motorcycle`)** | 155 (1/3 tập) | 76.13% | **76.51%** | **+0.38%** | Nâng mạnh Precision xe máy từ 75.73% lên **88.15%**, giảm báo động giả. |
| **Xe tải / Xe buýt (`truck-bus`)** | 147 (3/3 tập) | 57.15% | **70.79%** | **+13.64%** | **Cải thiện vượt bậc**: Nhận diện chính xác phương tiện cỡ lớn nguy hiểm nhất. |

### 2.4. Kết luận Thực nghiệm 1
- **YOLO26s** vượt trội áp đảo YOLO11s ở tất cả chỉ số tổng hợp: **Precision (81.62% vs 80.60%)**, **Recall (73.39% vs 64.42%)**, **mAP@0.5 (80.24% vs 70.14%)**, và **mAP@0.5:0.95 (61.87% vs 53.21%)**.
- YOLO26s thắng tuyệt đối trên **cả 5/5 phân lớp**, đặc biệt bứt phá ở `train` (**+27.12%**), `truck-bus` (**+13.64%**) và `person` (**+3.99%**).
- **YOLO26s chính thức được lựa chọn** làm kiến trúc mạng nơ-ron nhận diện nòng cốt cho toàn bộ hệ thống.

---

## 3. Thực nghiệm 2: Đánh giá & So sánh Giải thuật Tracking Đa Đối tượng (MOT)

### 3.1. Mục tiêu & Bài toán Thực tế
Sau khi lựa chọn mô hình nhận diện cốt lõi **`yolo26s`**, hệ thống cần thuật toán theo dõi đa đối tượng để:
- **Duy trì định danh (Track ID Consistency):** Tránh hiện tượng nhấp nháy, đổi ID liên tục khi vật thể di chuyển qua các khung hình.
- **Giám sát thời gian lưu trú trong vùng nguy hiểm (Dwell-Time Monitoring $\ge 3.0s$):** Phân biệt giữa người/xe qua đường hợp pháp với chướng ngại vật cố tình dừng đỗ hoặc kẹt xe trên đường ray.
- **Tối ưu chi phí tính toán (Overhead):** Đảm bảo thuật toán tracking chiếm ít thời gian CPU nhất có thể.

### 3.2. Bảng So sánh Hiệu năng Tracking Thực tế
*(Đo đạc trên cùng video kiểm thử `Dataset_test/test_exp.mp4`, 10 khung hình warmup, trích xuất từ `exp2_tracker_comparison/results/tracker_comparison.csv`)*

| Phương pháp | Độ trễ TB (Latency) | P95 Latency | Tốc độ (FPS) | Chi phí thêm (Overhead) | Số lượng ID | Tuổi thọ ID TB | Số lần đứt ID | Hỗ trợ Dwell-Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Pure YOLO (Không Tracking)** | 93.80 ms | 113.26 ms | **10.7 FPS** | **0.00 ms (Gốc)** | *NaN* | 0.0 frames | *NaN* | ❌ **Bất khả thi** (Không thể tính thời gian dừng đỗ) |
| **2. YOLO + BoT-SORT / DeepSORT** | 120.90 ms | 159.97 ms | 8.3 FPS | +27.10 ms | 5 | 88.0 frames | 7 | ⚠️ Kém tối ưu (Overhead cao, tốn tài nguyên Re-ID) |
| **3. YOLO + ByteTrack (Đề xuất)** | **113.01 ms** | **126.55 ms** | **8.8 FPS** | **+19.21 ms** | **5** | **88.0 frames** | **7** | ✅ **Tối ưu nhất** (Tiết kiệm 29.1% overhead so với BoT-SORT) |

### 3.3. Kết luận Thực nghiệm 2
- **ByteTrack** đạt cùng chất lượng theo dõi (duy trì đầy đủ 5 đối tượng với tuổi thọ trung bình 88.0 frames/ID) nhưng giảm độ trễ bổ sung từ **27.10 ms xuống còn 19.21 ms** (**tiết kiệm 29.1% chi phí tính toán** so với BoT-SORT).
- P95 latency của ByteTrack duy trì ở mức **126.55 ms** (so với 159.97 ms của BoT-SORT), giúp luồng xử lý không bị giật lag.
- ByteTrack tận dụng liên kết hộp bao 2 giai đoạn (bao gồm cả các box điểm tự tin thấp) mà **không cần nạp thêm mô hình trích xuất đặc trưng Re-ID nặng nề**.
- **ByteTrack được lựa chọn chính thức** làm giải thuật theo dõi đối tượng của hệ thống.

---

## 4. Thực nghiệm 3: Đánh giá Tăng tốc Phần cứng & Tối ưu hóa Suy luận (PyTorch vs Intel OpenVINO)

### 4.1. Quy trình Chuyển đổi & Phần cứng Thực nghiệm
Mô hình hoàn thiện `yolo26s` được chuyển đổi sang định dạng **Intel OpenVINO FP16** (`.xml` / `.bin`) thông qua công cụ `models/yolo26s/export_openvino.py` và lưu trữ tại `models/yolo26s/model_openvino`.

Thực nghiệm đánh giá năng lực triển khai hệ thống trên máy tính trạm/văn phòng phổ thông giá rẻ (\$400 – \$600) thay vì các máy chủ GPU đắt tiền (\$2,000+):
- **Phần cứng thử nghiệm:** 11th Gen Intel Core i5-1135G7 @ 2.40GHz (4 Cores, 8 Threads), đồ họa tích hợp **Intel Iris Xe Graphics (96 EUs)**, RAM 16GB.

### 4.2. Bảng Đo đạc Hiệu năng Tăng tốc Phần cứng
*(Đo đạc 10 warmup + 50 khung hình kiểm thử, trích xuất từ `exp3_hardware_acceleration/results/hardware_benchmark.csv`)*

| STT | Cấu hình Mô hình | Thiết bị Phần cứng | Độ trễ TB (ms) | Trung vị (ms) | P95 Latency (ms) | Tốc độ (FPS) | Hệ số Tăng tốc | Chuẩn Real-Time ($\ge 30$ FPS) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **PyTorch Framework (`.pt`)** | Intel CPU | 100.13 ms | 89.19 ms | 130.51 ms | 10.0 FPS | **1.00x (Gốc)** | ❌ Không đạt (Giật lag) |
| 2 | **Intel OpenVINO (`.xml`)** | Intel CPU | 249.29 ms | 239.31 ms | 330.78 ms | 4.0 FPS | 0.40x | ❌ Không đạt |
| 3 | **Intel OpenVINO (`.xml`)** | **Intel Iris Xe iGPU** | **28.16 ms** | **28.06 ms** | **29.19 ms** | **35.5 FPS** | **3.56x** | ✅ **Vượt chuẩn (Tối ưu nhất)** |

### 4.3. Kết luận Thực nghiệm 3
- Chạy PyTorch CPU truyền thống chỉ đạt **10.0 FPS** (độ trễ 100.13 ms), không thể đáp ứng yêu cầu giám sát cảnh báo tức thời.
- Tối ưu hóa qua **Intel OpenVINO trên Intel Iris Xe iGPU** giúp tăng tốc vượt bậc lên **35.5 FPS** (độ trễ giảm sâu về **28.16 ms**, tăng tốc **3.56 lần**).
- Việc kích hoạt trên **Intel Iris Xe iGPU (35.5 FPS)** giải phóng hoàn toàn năng lực tính toán của CPU, cho phép hệ thống đồng thời thực hiện decode RTSP, chạy ByteTrack, giám sát Dwell time và duy trì Web Server mượt mà không nghẽn cổ chai.

---

## 5. Tổng kết Kiến trúc Công nghệ Tối ưu của Hệ thống

### 5.1. Quy trình Nghiên cứu & Ra quyết định Kỹ thuật 4 Bước (Pipeline Overview)

```
[BƯỚC 1: THỰC NGHIỆM 1]
So sánh kiến trúc trên dataset_test (5 lớp): YOLO11s vs YOLO26s
(YOLO26s vượt trội: mAP50 80.24% vs 70.14%, Recall 73.39% vs 64.42%, mAP50-95 61.87% vs 53.21%)
                    │
                    ▼
          LỰA CHỌN KIẾN TRÚC YOLO26
                    │
                    ▼
[MÔ HÌNH HÓA ĐỐI TƯỢNG XÂM NHẬP (YOLO26s)]
4 lớp giám sát xâm nhập: person, motorcycle, car, truck-bus (train lưu thông hợp lệ)
                    │
                    ▼
[BƯỚC 2: THỰC NGHIỆM 2]
Đánh giá giải thuật Tracking (MOT) với detector YOLO26s
(Pure YOLO vs BoT-SORT/DeepSORT vs ByteTrack)
                    │
                    ▼
        LỰA CHỌN BYTETRACK (Overhead chỉ +19.21 ms, tiết kiệm 29.1% chi phí so với BoT-SORT)
                    │
                    ▼
[BƯỚC 3: TỐI ƯU HÓA ĐỒ THỊ TÍNH TOÁN]
Chạy models/yolo26s/export_openvino.py
Chuyển đổi train-railway-yolo26s-all.pt -> OpenVINO FP16 (models/yolo26s/model_openvino)
                    │
                    ▼
[BƯỚC 4: THỰC NGHIỆM 3]
Benchmark tăng tốc phần cứng thực tế trên máy trạm
(PyTorch CPU 10.0 FPS -> OpenVINO Intel Iris Xe iGPU 35.5 FPS, độ trễ 28.16 ms, tăng tốc 3.56 lần)
```

### 5.2. Sơ đồ Luồng Hoạt động Thời gian thực (Runtime Flow)

```
                  LUỒNG CAMERA ĐƯỜNG SẮT (RTSP / OBS / WEBCAM / VIDEO)
                                           │
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │  TIỀN XỬ LÝ & SUY LUẬN MÔ HÌNH HỌC SÂU NHẬN DIỆN  │
                 │        • Mô hình được chọn: YOLO26s              │
                 │        • Tối ưu hóa: Intel OpenVINO FP16         │
                 │        • Bộ gia tốc: Intel Iris Xe iGPU          │
                 │        • Hiệu năng: 35.5 FPS | Độ trễ 28.16 ms    │
                 └─────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │   BỘ THEO DÕI ĐA ĐỐI TƯỢNG (MULTI-OBJECT TRACK)  │
                 │        • Giải thuật: ByteTrack                   │
                 │        • Chi phí tính toán: +19.21 ms             │
                 │        • Quản lý: Duy trì định danh (Track ID)   │
                 └─────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │    ĐÁNH GIÁ VÙNG NGUY HIỂM & THỜI GIAN LƯU TRÚ   │
                 │        • Giao cắt đa giác nguy hiểm (ROI Polygon)│
                 │        • Giám sát dừng đỗ / kẹt xe (Dwell Time)  │
                 │        • Ngưỡng vi phạm: Dwell Duration ≥ 3.0s   │
                 └─────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                      PHÁT CHUÔNG CẢNH BÁO / LƯU VIDEO XÂM NHẬP / 
                        TRUYỀN PHÁT GIAO DIỆN WEB GIÁM SÁT REALTIME
```

---

*Toàn bộ mã nguồn, cấu hình và dữ liệu chi tiết của từng thực nghiệm được lưu trữ đầy đủ trong các thư mục con tương ứng (`exp1_model_comparison/`, `exp2_tracker_comparison/`, `exp3_hardware_acceleration/`).*
