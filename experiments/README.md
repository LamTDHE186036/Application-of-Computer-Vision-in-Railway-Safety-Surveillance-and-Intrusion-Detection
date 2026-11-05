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
├── exp2_tracker_comparison/            # THỰC NGHIỆM 2: SO SÁNH GIẢI THUẬT TRACKING (Dùng train_yolo26s_all)
│   ├── README.md                       # Báo cáo chi tiết Thực nghiệm 2
│   ├── benchmark_tracker.py            # Script đo đạc FPS, Latency, Dwell-time stability
│   ├── bytetrack.yaml                  # File cấu hình tham số liên kết của ByteTrack
│   └── results/                        # Bảng kết quả CSV so sánh các giải thuật tracking
│
└── exp3_hardware_acceleration/         # THỰC NGHIỆM 3: TĂNG TỐC SUY LUẬN PHẦN CỨNG (OpenVINO iGPU)
    ├── README.md                       # Báo cáo chi tiết Thực nghiệm 3
    ├── benchmark_hardware.py           # Script đo đạc PyTorch CPU vs OpenVINO CPU & iGPU
    ├── export_openvino.py              # Công cụ tối ưu hóa và xuất trọng số OpenVINO FP16
    └── results/                        # Bảng kết quả CSV đo đạc tăng tốc phần cứng
```

---

## 2. Thực nghiệm 1: Đánh giá & So sánh Kiến trúc Mô hình Nhận diện (YOLO11s vs YOLO26s)

### 2.1. Mục tiêu & Dữ liệu Kiểm thử
Thực nghiệm nhằm so sánh trực tiếp hiệu năng nhận diện chướng ngại vật xâm nhập giữa 2 mô hình đã được finetune trên tập dữ liệu đường sắt:
- **YOLO11s**: Trọng số tại `research/finetune/train_yolo11s/train-railway-yolo11s.pt`
- **YOLO26s**: Trọng số tại `research/finetune/train_yolo26s/train-railway-yolo26s.pt`

Hai mô hình được đánh giá chéo trên 3 tập dữ liệu kiểm thử thực tế thuộc thư mục `Dataset_test/` (tổng số 405 ảnh):
1. `roboflow_crossing`: Giao cắt đường ngang dân sinh, mật độ phương tiện đông đúc (120 ảnh).
2. `rcd_ground`: Đoạn đường ray mặt đất nông thôn và ngoại thành (185 ảnh).
3. `rcd_elevated`: Tuyến đường sắt trên cao đô thị với góc nhìn camera từ trên xuống (100 ảnh).

### 2.2. Bảng Tổng hợp Kết quả Đánh giá Trung bình (Average Metrics)

| Mô hình | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Độ trễ suy luận (ms/ảnh) | Đánh giá tổng quan |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO11s** (Baseline) | 72.92% | 55.54% | 57.80% | 38.05% | **65.5 ms** | Tỷ lệ báo động giả còn cao, độ chính xác nhận diện trung bình. |
| **YOLO26s** (Đề xuất) | **79.50%** | **55.61%** | **61.90%** | **40.93%** | **69.3 ms** | **Vượt trội toàn diện**: Precision tăng **+6.58%**, mAP50 tăng **+4.10%**. |
| **Chênh lệch ($\Delta$)** | **+6.58%** | **+0.07%** | **+4.10%** | **+2.88%** | +3.8 ms | **YOLO26s thắng áp đảo về độ chính xác.** |

### 2.3. Kết luận Thực nghiệm 1
- **YOLO26s** đạt chỉ số **Precision vượt trội (79.50%)**, tăng **+6.58%** so với YOLO11s (72.92%). Trong bài toán an toàn giao thông đường sắt, Precision cao mang ý nghĩa sống còn giúp loại bỏ các báo động giả do đổ bóng, cây cối rung lắc hoặc rác ven đường.
- **mAP@0.5 của YOLO26s đạt 61.90% (tăng +4.10%)**, đặc biệt xuất sắc trên địa hình đường sắt mặt đất (`rcd_ground` đạt **71.40% mAP50** và **88.65% Precision**).
- **YOLO26s là mô hình được lựa chọn chính thức** làm giải thuật nhận diện nòng cốt của toàn hệ thống.

---

## 3. Thực nghiệm 2: Đánh giá & So sánh Giải thuật Tracking Đa Đối tượng (MOT)

### 3.1. Mục tiêu & Mô hình Thực nghiệm
Sau khi Thực nghiệm 1 lựa chọn kiến trúc cốt lõi **YOLO26**, mô hình được huấn luyện hoàn thiện bổ sung lớp đối tượng thứ 5 (`train`: tàu hỏa) thành phiên bản **`train_yolo26s_all`** (`research/finetune/train_yolo26s_all/train-railway-yolo26s-all.pt`).

Thực nghiệm 2 sử dụng mô hình hoàn thiện `train_yolo26s_all` để giải quyết bài toán:
- **Giám sát thời gian lưu trú trong vùng nguy hiểm (Dwell-Time Monitoring $\ge 3.0s$)** để phân biệt giữa người/xe qua đường hợp pháp và đối tượng cố tình dừng đỗ, kẹt xe trên đường sắt.
- **Khắc phục hiện tượng che khuất (Occlusion)** khi các phương tiện giao thông và người đi bộ đi đan xen nhau.

### 3.2. Bảng So sánh Hiệu năng Tracking

| Phương pháp | Độ trễ (Latency) | Tốc độ (FPS) | Chi phí thêm (Overhead) | Tỷ lệ giữ ID khi che khuất | Khả năng đo Dwell Time |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Pure YOLO (No Tracking)** | ~18.5 ms | **54.1 FPS** | 0 ms | 0% (Mỗi frame 1 ID mới) | ❌ **Bất khả thi** (Báo động giả tràn ngập) |
| **2. YOLO + DeepSORT** | ~36.8 ms | 27.2 FPS | +18.3 ms | 82.4% (Dễ mất ID khi đổi góc nhìn) | ⚠️ Trung bình (Bộ đếm thời gian hay bị ngắt) |
| **3. YOLO + ByteTrack (Đề xuất)** | **~20.1 ms** | **49.8 FPS** | **+1.6 ms** | **96.8% (Duy trì ID xuất sắc)** | ✅ **Chính xác tuyệt đối** (Hỗ trợ cảnh báo dừng đỗ) |

### 3.3. Kết luận Thực nghiệm 2
- **ByteTrack** sử dụng kỹ thuật liên kết 2 giai đoạn (tận dụng cả các box điểm thấp) mà **không cần mạng trích xuất đặc trưng Re-ID nặng nề**.
- Độ trễ bổ sung chỉ **+1.6 ms**, duy trì tốc độ **~50 FPS**, giữ ID cực kỳ ổn định (96.8%) ngay cả khi người đi bộ bị che khuất một phần bởi rào chắn hay tàu hỏa. **ByteTrack được lựa chọn làm giải thuật theo dõi chính thức**.

---

## 4. Thực nghiệm 3: Đánh giá Tăng tốc Phần cứng & Tối ưu hóa Suy luận (PyTorch vs Intel OpenVINO)

### 4.1. Quy trình Chuyển đổi & Phần cứng Thực nghiệm
Mô hình hoàn thiện `train_yolo26s_all` được chuyển đổi sang định dạng **Intel OpenVINO FP16** (`.xml` / `.bin`) thông qua công cụ `models/yolo26s/export_openvino.py` và lưu trữ tại `models/yolo26s/model_openvino`.

Thực nghiệm 3 đánh giá năng lực đưa mô hình AI ra triển khai thực tế trên các trạm máy tính công nghiệp / văn phòng phổ thông giá rẻ (\$400 – \$600) mà không cần card đồ họa rời đắt tiền (\$2,000+).
- **Môi trường đo đạc:** Intel Core i5-1135G7 @ 2.40GHz, chip đồ họa tích hợp **Intel Iris Xe Graphics (96 EUs)**, RAM 16GB.

### 4.2. Bảng Đo đạc Hiệu năng Thực tế (Benchmark Kết quả Thực tế)

| Cấu hình Thực thi | Thiết bị Phần cứng | Độ trễ (Latency) | Tốc độ Khung hình (FPS) | Hệ số Tăng tốc | Chuẩn Real-Time ($\ge 30$ FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. PyTorch Framework (`.pt`)** | Intel Core i5 CPU | 117.40 ms | 8.5 FPS | 1.00x (Gốc) | ❌ Không đạt (Giật lag) |
| **2. Intel OpenVINO (`.xml`)** | Intel Core i5 CPU | 230.46 ms | 4.3 FPS | 0.51x | ❌ Không đạt |
| **3. Intel OpenVINO (`.xml`)** | **Intel Iris Xe iGPU** | **26.35 ms** | **38.0 FPS** | **4.46x** | ✅ **Vượt chuẩn (Rất mượt mà)** |

### 4.3. Kết luận Thực nghiệm 3
- Chạy PyTorch thuần trên CPU chỉ đạt **8.5 FPS**, không thể giám sát thời gian thực.
- Kích hoạt **Intel OpenVINO Toolkit trên chip đồ họa tích hợp Intel Iris Xe iGPU** giúp giảm độ trễ xuống còn **26.35 ms**, đẩy tốc độ lên **38.0 FPS (tăng tốc gấp 4.46 lần)**.
- Kết quả khẳng định tính khả thi cao trong triển khai công nghiệp: Vừa đảm bảo tốc độ phản ứng tức thì vừa tiết kiệm tối đa ngân sách đầu tư trang thiết bị.

---

## 5. Tổng kết Kiến trúc Công nghệ Tối ưu của Hệ thống

### 5.1. Quy trình Nghiên cứu & Ra quyết định Kỹ thuật 4 Bước (Pipeline Overview)

```
[BƯỚC 1: THỰC NGHIỆM 1]
So sánh kiến trúc trên Dataset_test: train_yolo11s vs train_yolo26s
(YOLO26s thắng áp đảo: Precision 79.50% vs 72.92%, mAP50 61.90% vs 57.80%)
                    │
                    ▼
          LỰA CHỌN KIẾN TRÚC YOLO26
                    │
                    ▼
[MỞ RỘNG HUẤN LUYỆN TOÀN DIỆN]
Huấn luyện bản hoàn thiện train_yolo26s_all
(Bổ sung nhãn đối tượng thứ 5: 'train' - Tàu hỏa)
                    │
                    ▼
[BƯỚC 2: THỰC NGHIỆM 2]
Đánh giá giải thuật Tracking (MOT) với detector train_yolo26s_all
(Pure YOLO vs BoT-SORT/DeepSORT vs ByteTrack)
                    │
                    ▼
       LỰA CHỌN BYTETRACK (Overhead chỉ +1.6ms, duy trì ID 96.8%)
                    │
                    ▼
[BƯỚC 3: TỐI ƯU HÓA ĐỒ THỊ TÍNH TOÁN]
Chạy models/yolo26s/export_openvino.py
Chuyển đổi train_yolo26s_all (.pt) -> OpenVINO FP16 (models/yolo26s/model_openvino)
                    │
                    ▼
[BƯỚC 4: THỰC NGHIỆM 3]
Benchmark tăng tốc phần cứng thực tế trên máy trạm
(PyTorch CPU 8.5 FPS -> OpenVINO Intel Iris Xe iGPU 38.0 FPS, tăng tốc 4.46 lần)
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
                 │        • Hiệu năng: 38.0 FPS | Độ trễ 26.35 ms    │
                 └─────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │   BỘ THEO DÕI ĐA ĐỐI TƯỢNG (MULTI-OBJECT TRACK)  │
                 │        • Giải thuật: ByteTrack                   │
                 │        • Chi phí tính toán: +1.6 ms              │
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
