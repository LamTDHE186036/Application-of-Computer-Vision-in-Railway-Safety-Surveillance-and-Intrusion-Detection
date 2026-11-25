# THỰC NGHIỆM 3: ĐÁNH GIÁ TỐI ƯU HÓA PHẦN CỨNG & TĂNG TỐC SUY LUẬN (YOLO26s & YOLO26m)

## 1. Đặt vấn đề & Mục tiêu Nghiên cứu

Trong triển khai thực tế hệ thống Giám sát An toàn Đường sắt:
- **Rào cản chi phí phần cứng biên (Edge Computing Constraints):** Trang bị máy chủ gắn GPU rời chuyên dụng (NVIDIA RTX / Tesla) tại mọi điểm gác đường ngang tốn kém (`\$1,500 – \$3,000`/trạm), tiêu thụ điện năng cao (`120W – 250W`) và khó vận hành bền bỉ 24/7 trong tủ kỹ thuật ngoài trời.
- **Yêu cầu thời gian thực (Real-Time Standard):** Luồng camera giám sát công nghiệp chuẩn PAL/CCTV vận hành ở mức **`25 – 30 FPS`** (tương ứng độ trễ xử lý $\le 33 – 40\text{ ms/frame}$).
- **Giải pháp tiếp cận:** Chuyển đổi cả 2 mô hình **`YOLO26s`** (`models/yolo26s/export_openvino.py`) và **`YOLO26m`** (`models/yolo26m/export_openvino.py`) sang định dạng **Intel OpenVINO™ IR FP16** (`.xml` / `.bin`) và đo đạc trực tiếp trên máy tính văn phòng không GPU rời nhằm tìm ranh giới tối ưu cho cơ chế **Adaptive Hardware-Aware Model Selection (v1.3.0)**.

Thực nghiệm so sánh **6 cấu hình** (`{YOLO26s, YOLO26m} × {PyTorch CPU, OpenVINO FP16 CPU, OpenVINO FP16 Intel Iris Xe iGPU}`).

---

## 2. Thông số Phần cứng & Quy trình Đo đạc

- **CPU:** 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4 Cores, 8 Threads).
- **Đồ họa tích hợp (iGPU):** Intel(R) Iris(R) Xe Graphics (96 EUs).
- **GPU rời (NVIDIA CUDA):** Không có.
- **RAM / HĐH:** 16 GB DDR4 Dual-Channel / Windows 11 64-bit.
- **Dữ liệu đo đạc:** `50` khung hình thực tế nạp sẵn vào RAM từ `dataset_test/test_exp.mp4` (`25.0 FPS`), kích thước suy luận `640×640`, chạy trước `10` khung hình **Warmup** cho mỗi cấu hình để loại trừ độ trễ biên dịch JIT/OpenCL ban đầu.

---

## 3. Cơ chế Kỹ thuật Tối ưu hóa Intel OpenVINO FP16

1. **Layer Fusion (Gộp toán tử):** Gộp chuỗi `Conv2d + BatchNorm2d + SiLU` thành một hạt nhân toán tử duy nhất (giảm `YOLO26s` xuống `120` layers / `20.8 GFLOPs` và `YOLO26m` xuống `130` layers / `68.1 GFLOPs`).
2. **Nén trọng số chuẩn bán chính xác FP16:** Giảm một nửa băng thông bộ nhớ và tận dụng nhân tính toán FP16 phần cứng trên 96 EUs của Intel Iris Xe iGPU.
3. **Giải phóng tải CPU (iGPU Offloading):** Chuyển toàn bộ tính toán mạng nơ-ron sang iGPU để CPU rảnh tài nguyên cho giải mã video, chạy ByteTrack, kiểm tra ROI/Dwell Time, phục vụ Flask MJPEG và chuyển mã FFmpeg H.264.

---

## 4. Kết quả Đo đạc Thực tế
*(Trích xuất trực tiếp từ `results/hardware_benchmark.csv`, chạy lệnh `python experiments/exp3_hardware_acceleration/benchmark_hardware.py --frames 50`)*

| Mô hình | Cấu hình thực thi | Thiết bị | Độ trễ TB (ms) | Trung vị (ms) | P95 Latency (ms) | Min / Max (ms) | Tốc độ (FPS) | Tăng tốc | Chuẩn Real-Time ($\ge 25$ FPS) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`YOLO26s`** | PyTorch (`.pt`) | Intel CPU | 104.77 ms | 98.69 ms | 126.61 ms | 88.81 / 161.66 ms | 9.5 FPS | 1.00x (Gốc) | ❌ Không đạt |
| **`YOLO26s`** | OpenVINO FP16 (`.xml`) | Intel CPU | 270.90 ms | 267.70 ms | 383.94 ms | 177.00 / 462.13 ms | 3.7 FPS | 0.39x | ❌ Không đạt |
| **`YOLO26s`** | **OpenVINO FP16 (`.xml`)** | **Intel Iris Xe iGPU** | **32.51 ms** | **32.34 ms** | **34.69 ms** | **28.99 / 38.25 ms** | **30.8 FPS** | **3.22x** | ✅ **Đạt chuẩn Real-Time** |
| **`YOLO26m`** | PyTorch (`.pt`) | Intel CPU | 288.49 ms | 286.00 ms | 304.97 ms | 273.75 / 337.21 ms | 3.5 FPS | 1.00x (Gốc) | ❌ Không đạt |
| **`YOLO26m`** | OpenVINO FP16 (`.xml`) | Intel CPU | 281.31 ms | 87.73 ms | 739.30 ms | 76.95 / 1044.94 ms | 3.6 FPS | 1.03x | ❌ Không đạt |
| **`YOLO26m`** | **OpenVINO FP16 (`.xml`)** | **Intel Iris Xe iGPU** | **86.85 ms** | **86.57 ms** | **93.21 ms** | **82.86 / 95.93 ms** | **11.5 FPS** | **3.32x** | ❌ **Chưa đạt trên iGPU** |

---

## 5. Phân tích Khoa học & Ý nghĩa Kiến trúc

1. **Intel Iris Xe iGPU tăng tốc đồng đều `3.22x – 3.32x` cho cả 2 mô hình:**
   - **`YOLO26s`:** Rút độ trễ từ `104.77 ms` (`9.5 FPS`) xuống **`32.51 ms`** (**`30.8 FPS`**, P95 chỉ `34.69 ms` — cực kỳ ổn định), vượt chuẩn Real-Time cả `25 FPS` lẫn `30 FPS`.
   - **`YOLO26m`:** Rút độ trễ từ `288.49 ms` (`3.5 FPS`) xuống **`86.85 ms`** (**`11.5 FPS`**, tăng tốc `3.32x`), tuy nhiên do khối lượng tính toán `68.1 GFLOPs` gấp `3.27x` so với `YOLO26s` (`20.8 GFLOPs`), tốc độ `11.5 FPS` trên iGPU văn phòng chưa đủ mượt để chạy Real-Time 25–30 FPS.
2. **Vì sao OpenVINO FP16 chạy trên Intel CPU chậm hơn PyTorch CPU (`3.7 FPS` vs `9.5 FPS` ở `YOLO26s`)?**
   - Mô hình OpenVINO được nén chuẩn **`FP16`** để tối ưu cho nhân phần cứng FP16 trên Intel Iris Xe iGPU. Trên CPU thế hệ 11 (Tiger Lake) không có tập lệnh FP16 gốc (`AVX-512 FP16` chỉ xuất hiện từ Sapphire Rapids), OpenVINO phải chuyển đổi kiểu dữ liệu ngược lên FP32 khi tính toán trên CPU, gây phát sinh chi phí chuyển đổi. Do đó khi chạy OpenVINO FP16, bắt buộc nên kích hoạt đích phần cứng `intel:gpu`.
3. **Cơ sở vững chắc cho chiến lược Chọn Mô hình Thích ứng Phần cứng (`v1.3.0`):**
   - **Máy văn phòng / Mini-PC không GPU rời (`\$400 – \$600`, TDP `15–28W`):** Hệ thống tự động chọn **`YOLO26s` OpenVINO FP16 trên Intel Iris Xe iGPU (`30.8 FPS`)**, đảm bảo vận hành liên tục 24/7 mượt mà không rớt frame.
   - **Máy trạm có GPU rời NVIDIA CUDA:** Hệ thống tự động chọn **`YOLO26m` PyTorch CUDA** để tận hưởng độ chính xác cao nhất (`85.96%` mAP@0.5, `72.47%` mAP@0.5:0.95) mà vẫn dư thừa tốc độ Real-Time.

---

## 6. Hướng dẫn Xuất Mô hình & Tái hiện Thực nghiệm

```bash
# 1. Xuất mô hình sang Intel OpenVINO FP16
python models/yolo26s/export_openvino.py
python models/yolo26m/export_openvino.py

# 2. Chạy lại benchmark phần cứng cho cả 2 mô hình (10 warmup + 50 frames)
python experiments/exp3_hardware_acceleration/benchmark_hardware.py --frames 50
```

Toàn bộ dữ liệu thô và biểu đồ được lưu tại `results/hardware_benchmark.csv` và `results/chart_hardware_benchmark.png`.
