# BÁO CÁO NGHIÊN CỨU THỰC NGHIỆM VÀ ĐÁNH GIÁ HỆ THỐNG (v1.3.0)
## Ứng dụng Thị giác Máy tính trong Giám sát An toàn Đường sắt và Phát hiện Xâm nhập

Tài liệu này tổng hợp toàn bộ phương pháp luận, quy trình triển khai và số liệu thực nghiệm đo đạc thực tế của **05 bài toán thực nghiệm cốt lõi** trong hệ thống **RailSafe-CV v1.3.0** (tích hợp trực tiếp vào **Chương 4 — Validation** của báo cáo đồ án tốt nghiệp).

---

## MỤC LỤC BÁO CÁO THỰC NGHIỆM

1. [Cấu trúc Thư mục Thực nghiệm](#1-cấu-trúc-thư-mục-thực-nghiệm)
2. [Thực nghiệm 1: So sánh 3 Kiến trúc Mô hình Nhận diện (YOLO11s vs YOLO26s vs YOLO26m)](#2-thực-nghiệm-1-so-sánh-3-kiến-trúc-mô-hình-nhận-diện-yolo11s-vs-yolo26s-vs-yolo26m)
3. [Thực nghiệm 2: Đánh giá Giải thuật Tracking Đa Đối tượng (YOLO26s & YOLO26m)](#3-thực-nghiệm-2-đánh-giá-giải-thuật-tracking-đa-đối-tượng-yolo26s--yolo26m)
4. [Thực nghiệm 3: Tăng tốc Phần cứng & Tối ưu hóa Suy luận (YOLO26s & YOLO26m)](#4-thực-nghiệm-3-tăng-tốc-phần-cứng--tối-ưu-hóa-suy-luận-yolo26s--yolo26m)
5. [Thực nghiệm 4: Đánh giá Tái lập YOLO26m trên Bộ Test 119 Ảnh (COCO Pretrained vs Fine-tune)](#5-thực-nghiệm-4-đánh-giá-tái-lập-yolo26m-trên-bộ-test-119-ảnh-coco-pretrained-vs-fine-tune)
6. [Thực nghiệm 5: Đánh giá Hiệu năng Thời gian chạy Luồng Web & Phân tích Ca Báo sót](#6-thực-nghiệm-5-đánh-giá-hiệu-năng-thời-gian-chạy-luồng-web--phân-tích-ca-báo-sót)
7. [Tổng kết Kiến trúc Chọn Mô hình Thích ứng Phần cứng (v1.3.0)](#7-tổng-kết-kiến-trúc-chọn-mô-hình-thích-ứng-phần-cứng-v130)

---

## 1. Cấu trúc Thư mục Thực nghiệm

```text
experiments/
├── README.md                           # Báo cáo tổng hợp toàn bộ 5 thực nghiệm (v1.3.0)
│
├── exp1_model_comparison/              # THỰC NGHIỆM 1: SO SÁNH 3 MÔ HÌNH (YOLO11s vs YOLO26s vs YOLO26m)
│   ├── README.md                       # Báo cáo chi tiết Thực nghiệm 1
│   ├── evaluate_models.py              # Script kiểm thử 3 mô hình trên dataset_test (MappedDetectionValidator)
│   └── results/                        # 5 bảng CSV, 3 biểu đồ tổng hợp + 9 thư mục PR/F1/Confusion Matrix
│
├── exp2_tracker_comparison/            # THỰC NGHIỆM 2: SO SÁNH TRACKING MOT (YOLO26s & YOLO26m)
│   ├── README.md                       # Báo cáo chi tiết Thực nghiệm 2
│   ├── benchmark_tracker.py            # Script đo Latency, P95, Overhead, FPS, Lifespan cho cả 2 mô hình
│   ├── bytetrack.yaml                  # Cấu hình tham số ByteTrack
│   └── results/                        # tracker_comparison.csv & chart_tracker_comparison.png
│
├── exp3_hardware_acceleration/         # THỰC NGHIỆM 3: TĂNG TỐC PHẦN CỨNG (YOLO26s & YOLO26m)
│   ├── README.md                       # Báo cáo chi tiết Thực nghiệm 3
│   ├── benchmark_hardware.py           # Script benchmark PyTorch CPU vs OpenVINO CPU & Iris Xe iGPU
│   └── results/                        # hardware_benchmark.csv & chart_hardware_benchmark.png
│
├── exp4_yolo26m_thesis/                # THỰC NGHIỆM 4: ĐÁNH GIÁ TÁI LẬP YOLO26m (COCO vs FINE-TUNE, 119 ẢNH)
│   ├── README.md                       # Hướng dẫn tái lập và tổng hợp kết quả đánh giá thesis
│   ├── score.py                        # Script chấm khóa SHA-256 checkpoint & ProjectedClassifier 5 lớp
│   ├── image_metrics.py                # Hàm tính AP50, AP50-95 và Precision/Recall ngưỡng cố định
│   └── plot_confusion_matrix.py        # Script vẽ ma trận nhầm lẫn so sánh 2 mô hình
│
└── exp5_web_pipeline_evaluation/       # THỰC NGHIỆM 5: HIỆU NĂNG LUỒNG WEB & PHÂN TÍCH CA BÁO SÓT
    ├── README.md                       # Báo cáo chi tiết Thực nghiệm 5 (Mục 4.3.4 & 4.3.5)
    ├── evaluate_web_pipeline.py        # Script phân rã độ trễ toàn luồng Web & chẩn đoán ca báo sót ROI
    ├── evaluate_events_offline.py      # Bộ chấm điểm cấp độ sự kiện ngoại tuyến (tIoU, Event P/R/F1, FAR, Delay)
    └── results/                        # web_pipeline_runtime.csv & web_pipeline_diagnostics.json
```

---

## 2. Thực nghiệm 1: So sánh 3 Kiến trúc Mô hình Nhận diện (YOLO11s vs YOLO26s vs YOLO26m)

### 2.1. Dữ liệu & Phương pháp Kiểm định
Đánh giá chéo độc lập **3 mô hình 5 lớp** (`car`, `motorcycle`, `person`, `truck-bus`, `train`) trên 3 tập dữ liệu kiểm thử thực tế thuộc `dataset_test/` (tổng **404 ảnh / 1.238 đối tượng**: `roboflow_crossing`, `rcd_ground`, `rcd_elevated`) qua bộ kiểm định `MappedDetectionValidator` ánh xạ 1-1 theo tên lớp.

### 2.2. Bảng Tổng hợp Macro Average trên 3 Tập Kiểm thử Độc lập
*(Trích xuất từ `exp1_model_comparison/results/comparison_average.csv`, `batch=1`, `imgsz=640`, `device=cpu`)*

| Mô hình | Dung lượng | Params / GFLOPs | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 | Suy luận CPU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`YOLO11s`** *(Baseline)* | 18.29 MB | 9.43M / 21.68G | 80.60% | 64.42% | 71.30% | 70.14% | 53.21% | 89.33 ms |
| **`YOLO26s`** *(Edge Real-Time)* | **19.38 MB** | **9.95M / 22.76G** | 81.62% | 73.39% | 76.89% | 80.24% | 61.87% | **88.59 ms** |
| **`YOLO26m`** *(High-Accuracy)* | 42.03 MB | 21.78M / 74.99G | **83.16%** | **82.06%** | **82.58%** | **85.96%** | **72.47%** | 257.84 ms |

### 2.3. Hiệu năng Trung bình theo Từng Lớp (5 Lớp — mAP@0.5 / mAP@0.5:0.95)
*(Trích xuất từ `exp1_model_comparison/results/comparison_per_class_avg.csv`)*

| Lớp Đối tượng | Targets | `YOLO11s` | `YOLO26s` | `YOLO26m` |
| :--- | :---: | :---: | :---: | :---: |
| **`car`** (Ô tô) | 613 (3/3) | 79.91% / 64.05% | 82.40% / 65.59% | **90.78% / 76.90%** |
| **`motorcycle`** (Xe máy) | 155 (1/3) | 76.13% / 47.54% | 76.51% / 49.93% | **81.09% / 54.56%** |
| **`person`** (Người) | 306 (3/3) | 83.69% / 56.67% | 87.68% / 60.71% | **91.68% / 70.58%** |
| **`train`** (Tàu hỏa) | 17 (2/3) | 53.94% / 43.20% | 81.06% / 62.78% | **81.52% / 76.53%** |
| **`truck-bus`** (Xe tải/Buýt) | 147 (3/3) | 57.15% / 45.43% | 70.79% / 57.80% | **79.25% / 67.97%** |

### 2.4. Kết luận Thực nghiệm 1
- **`YOLO26s`** vượt trội **`YOLO11s`** toàn diện (`+10.10%` mAP@0.5, `+8.97%` Recall) ở cùng phân khúc kích thước và tốc độ.
- **`YOLO26m`** nâng tiếp độ chính xác lên đỉnh cao (`85.96%` mAP@0.5, `72.47%` mAP@0.5:0.95 — tăng `+10.60%` mAP50-95 so với `YOLO26s` và giảm sâu sụt giảm Domain Shift xuống `-9.85%`), nhưng nặng hơn `2.17x` dung lượng và `3.29x` GFLOPs.

---

## 3. Thực nghiệm 2: Đánh giá Giải thuật Tracking Đa Đối tượng (YOLO26s & YOLO26m)

*(Đo trên 150 khung hình thực tế của `dataset_test/test_exp.mp4` với OpenVINO FP16 trên Intel Iris Xe iGPU, trích từ `exp2_tracker_comparison/results/tracker_comparison.csv`)*

| Mô hình | Phương pháp | Độ trễ TB | P95 Latency | Overhead | Tốc độ (FPS) | Số ID | Vòng đời TB | Hỗ trợ Dwell Time (mặc định 5.0s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`YOLO26s`** | Pure YOLO | 29.27 ms | 32.46 ms | 0.00 ms | 34.2 FPS | N/A | 0.0 frames | ❌ Bất khả thi |
| | BoT-SORT | 33.94 ms | 35.65 ms | +4.67 ms | 29.5 FPS | 2 | 126.0 frames | ⚠️ Tốn thêm GMC optical flow |
| | **ByteTrack (Đề xuất)** | **29.54 ms** | **30.64 ms** | **+0.27 ms** | **33.9 FPS** | **2** | **126.0 frames** | ✅ **Tối ưu (-94.2% overhead)** |
| **`YOLO26m`** | Pure YOLO | 75.63 ms | 86.35 ms | 0.00 ms | 13.2 FPS | N/A | 0.0 frames | ❌ Bất khả thi |
| | BoT-SORT | 87.06 ms | 117.26 ms | +11.43 ms | 11.5 FPS | 2 | 129.5 frames | ⚠️ P95 tăng mạnh |
| | **ByteTrack (Đề xuất)** | **75.17 ms** | **81.30 ms** | **≈ 0 ms** | **13.3 FPS** | **2** | **129.5 frames** | ✅ **Tối ưu (nhanh hơn 11.89 ms)** |

- **Kết luận:** **ByteTrack** là bộ theo dõi tối ưu tuyệt đối trên cả 2 mô hình (đạt cùng chất lượng giữ ID nhưng nhẹ hơn BoT-SORT `4.40 ms` trên `YOLO26s` và `11.89 ms` trên `YOLO26m`).

---

## 4. Thực nghiệm 3: Tăng tốc Phần cứng & Tối ưu hóa Suy luận (YOLO26s & YOLO26m)

*(Đo 10 warmup + 50 khung hình thực tế từ `test_exp.mp4` trên Intel Core i5-1135G7 + Intel Iris Xe Graphics 96 EUs, RAM 16GB, trích từ `exp3_hardware_acceleration/results/hardware_benchmark.csv`)*

| Mô hình | Cấu hình | Thiết bị | Độ trễ TB | Trung vị | P95 Latency | Tốc độ (FPS) | Tăng tốc | Chuẩn Real-Time ($\ge 25$ FPS) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`YOLO26s`** | PyTorch (`.pt`) | Intel CPU | 104.77 ms | 98.69 ms | 126.61 ms | 9.5 FPS | 1.00x | ❌ Không đạt |
| **`YOLO26s`** | OpenVINO FP16 (`.xml`) | Intel CPU | 270.90 ms | 267.70 ms | 383.94 ms | 3.7 FPS | 0.39x | ❌ Không đạt |
| **`YOLO26s`** | **OpenVINO FP16 (`.xml`)** | **Intel Iris Xe iGPU** | **32.51 ms** | **32.34 ms** | **34.69 ms** | **30.8 FPS** | **3.22x** | ✅ **Đạt chuẩn Real-Time** |
| **`YOLO26m`** | PyTorch (`.pt`) | Intel CPU | 288.49 ms | 286.00 ms | 304.97 ms | 3.5 FPS | 1.00x | ❌ Không đạt |
| **`YOLO26m`** | OpenVINO FP16 (`.xml`) | Intel CPU | 281.31 ms | 87.73 ms | 739.30 ms | 3.6 FPS | 1.03x | ❌ Không đạt |
| **`YOLO26m`** | **OpenVINO FP16 (`.xml`)** | **Intel Iris Xe iGPU** | **86.85 ms** | **86.57 ms** | **93.21 ms** | **11.5 FPS** | **3.32x** | ❌ **Chưa đạt trên iGPU** |

- **Kết luận:** OpenVINO FP16 trên Intel Iris Xe tăng tốc **`3.22x – 3.32x`** cho cả 2 mô hình. Tuy nhiên trên máy văn phòng không GPU rời, chỉ **`YOLO26s` OpenVINO iGPU (`30.8 FPS`)** đạt chuẩn Real-Time mượt mà, còn **`YOLO26m` (`11.5 FPS` trên iGPU)** phù hợp nhất với máy trạm trang bị **GPU rời NVIDIA CUDA**.

---

## 5. Thực nghiệm 4: Đánh giá Tái lập YOLO26m trên Bộ Test 119 Ảnh (COCO Pretrained vs Fine-tune)

*(So sánh có kiểm chứng SHA-256 giữa `YOLO26m gốc COCO` và `YOLO26m fine-tune epoch 13` trên 119 ảnh / 707 hộp nhãn thuộc `roboflow_crossing`, 5 lớp `car`, `motorcycle`, `person`, `truck-bus`, `train`, trích từ [exp4_yolo26m_thesis/README.md](exp4_yolo26m_thesis/README.md))*

| Mô hình | mAP@0.5 | mAP@0.5:0.95 | Precision (micro, conf 0.35) | Recall (micro, conf 0.35) |
| :--- | :---: | :---: | :---: | :---: |
| `YOLO26m` gốc COCO *(Projected 5 lớp)* | 82.46% | 64.51% | **83.67%** | 65.21% |
| **`YOLO26m` fine-tune epoch 13** | **86.11%** | **65.54%** | 79.57% | **73.27%** |

- **Kết luận:** Fine-tune trên miền đường ngang nâng mạnh **`+3.65%` mAP@0.5**, **`+1.03%` mAP@0.5:0.95** và đặc biệt tăng **`+8.06%` Recall (micro)** ở ngưỡng cố định `conf = 0.35, IoU = 0.50`, giảm đáng kể tình trạng bỏ sót đối tượng trong kịch bản giám sát an toàn.

---

## 6. Thực nghiệm 5: Đánh giá Hiệu năng Thời gian chạy Luồng Web & Phân tích Ca Báo sót

*(Đo trực tiếp chuỗi xử lý luồng Web hoàn chỉnh trên video đường ngang thực tế `demo1` — $1272 \times 720$, $30\ \mathrm{FPS}$, $774$ khung hình đo sau $20$ khung khởi động với `YOLO26s` OpenVINO FP16 trên Intel Iris Xe iGPU, trích từ [exp5_web_pipeline_evaluation/README.md](exp5_web_pipeline_evaluation/README.md))*

| Công đoạn trên luồng Web | Trung bình (ms) | Trung vị (ms) | P95 (ms) | Tỷ trọng |
| :--- | :---: | :---: | :---: | :---: |
| 1. Đọc khung hình (`cap.read`) | 2.12 | 1.95 | 2.92 | 5.4% |
| 2. Suy luận + Tracking (`model.track`) | 31.47 | 30.99 | 34.84 | 79.7% |
| 3. Xử lý ROI + Cảnh báo + Vẽ phủ (`process_and_draw`) | 3.27 | 3.11 | 4.30 | 8.3% |
| 4. Mã hóa MJPEG (`JPEG quality=80`) | 2.60 | 2.48 | 3.35 | 6.6% |
| **Tổng toàn luồng Web** | **39.47** | **38.79** | **43.79** | **100.0% (25.3 FPS)** |

- **Chẩn đoán ca báo sót / lọc bỏ trên ROI:** Ghi nhận **13 khoảng mất hộp bao ngắn** (1–7 khung hình, dài nhất `Motorcycle ID:38` 7 khung / 0.23 s do chói đèn pha & nhòe chuyển động), **0 ca đổi ID** nhờ `track_buffer = 30` của ByteTrack, **2 quỹ đạo `< 0.5 s`** không kích hoạt báo động đỏ và **1 quỹ đạo `0.5–1.0 s`** (`Person ID:10` 0.83 s) bị lọc bỏ khi chốt sự kiện theo ngưỡng `MIN_EVENT_DURATION = 1.0 s`. Đi kèm bộ công cụ đánh giá cấp sự kiện ngoại tuyến `evaluate_events_offline.py` (`tIoU`, `Event Precision/Recall/F1`, `FAR`, `Alert Detection Delay`).

---

## 7. Tổng kết Kiến trúc Chọn Mô hình Thích ứng Phần cứng (v1.3.0)

Từ bằng chứng định lượng của chuỗi thực nghiệm trên, hệ thống **RailSafe-CV v1.3.0** triển khai cơ chế **Adaptive Hardware-Aware Model Selection** trong `app/services/detector.py`:

```text
                    KHỞI ĐỘNG HỆ THỐNG RAILSAFE-CV v1.3.0
                                      │
                                      ▼
                Kiểm tra tài nguyên phần cứng (check_system_resources)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼ (Có NVIDIA CUDA GPU)                          ▼ (Không có GPU rời: Intel iGPU / CPU)
  ┌───────────────────────────────────┐          ┌──────────────────────────────────────────┐
  │         KÍCH HOẠT YOLO26m         │          │           KÍCH HOẠT YOLO26s              │
  │ • Ưu tiên: OpenVINO FP16          │          │ • Ưu tiên: OpenVINO FP16 (Intel iGPU)    │
  │ • Dự phòng khi lỗi OV: .pt (CUDA) │          │ • Dự phòng khi lỗi OV: .pt (CPU)         │
  │ • mAP@0.5 = 85.96% | 50-95=72.47% │          │ • 30.8 FPS (32.51 ms) | mAP@0.5 = 80.24% │
  └───────────────────┬───────────────┘          └─────────────────────┬────────────────────┘
                      └───────────────────────┬────────────────────────┘
                                              ▼
                       ByteTrack MOT (+0.27 ms overhead) -> Polygon ROI
                    -> Dual-Mode Dwell Time (0.5s / 5.0s) -> Black-box & Spotlight
```
