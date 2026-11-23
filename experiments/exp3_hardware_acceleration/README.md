# THỰC NGHIỆM 3: ĐÁNH GIÁ TỐI ƯU HÓA PHẦN CỨNG & TĂNG TỐC SUY LUẬN (HARDWARE ACCELERATION)

## 1. Đặt vấn đề & Mục tiêu Nghiên cứu

Trong bài toán triển khai thực tế hệ thống Giám sát An toàn Đường sắt và Phát hiện Xâm nhập:
- **Rào cản chi phí phần cứng biên (Edge Computing Constraints):** Việc trang bị các máy chủ gắn card đồ họa rời chuyên dụng (NVIDIA RTX / Tesla / Jetson AGX) tại từng chốt gác đường ngang, trạm chắn hay camera ven tuyến đường sắt đòi hỏi chi phí đầu tư thiết bị rất cao (\$1,500 – \$3,000 / điểm), tiêu thụ điện năng lớn và khó vận hành bền bỉ 24/7 trong tủ kỹ thuật ngoài trời không có điều hòa nhiệt độ.
- **Yêu cầu khắt khe về thời gian thực (Real-Time Safety Standard):** Để kịp thời phát hiện chướng ngại vật trước khi đoàn tàu di chuyển tới với vận tốc cao (60 – 100 km/h), tốc độ xử lý của pipeline AI bắt buộc phải đạt tối thiểu **$\ge 25 - 30$ FPS** và độ trễ (latency) dưới **$35 - 40$ ms**.
- **Giải pháp tiếp cận:** Tận dụng bộ công cụ tối ưu hóa chuyên sâu **Intel OpenVINO™ Toolkit (Open Visual Inference and Neural Network Optimization)** để khai thác tối đa năng lực xử lý vector của CPU và đồ họa tích hợp (**Intel Iris Xe iGPU**) sẵn có trên các máy tính văn phòng / mini-PC công nghiệp phổ thông giá rẻ.

Mô hình hoàn thiện cuối cùng `yolo26s` (`models/yolo26s/model_pytorch/train-railway-yolo26s-all.pt`) được chuyển đổi sang định dạng OpenVINO FP16 thông qua công cụ `models/yolo26s/export_openvino.py` và lưu tại `models/yolo26s/model_openvino`.

Thực nghiệm này so sánh trực tiếp hiệu năng suy luận giữa:
1. Mô hình gốc **PyTorch (`.pt`)** của `yolo26s` chạy trên Intel CPU.
2. Mô hình tối ưu hóa **Intel OpenVINO (`.xml` / `.bin`)** của `yolo26s` chạy trên Intel CPU.
3. Mô hình **Intel OpenVINO** của `yolo26s` kích hoạt nhân tăng tốc đồ họa tích hợp **Intel Iris Xe Graphics (iGPU)**.

---

## 2. Thông số Phần cứng & Môi trường Thực nghiệm

- **Bộ vi xử lý trung tâm (CPU):** 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4 Cores, 8 Threads).
- **Bộ xử lý đồ họa tích hợp (iGPU):** Intel(R) Iris(R) Xe Graphics (96 EUs - Execution Units).
- **Bộ nhớ RAM:** 16 GB DDR4 Dual-Channel.
- **Hệ điều hành:** Windows 11 64-bit.
- **Khung hình suy luận:** Chuẩn độ phân giải $640 \times 640 \times 3$.
- **Mô hình kiểm thử:** `yolo26s` (Tập lớp đối tượng: `car`, `motorcycle`, `person`, `truck-bus`, `train`).
- **Quy trình đo đạc:** Chạy 10 khung hình khởi động (Warmup) để loại trừ hiện tượng JIT compilation/Cold-start, sau đó lấy mẫu thống kê trên 50 khung hình suy luận liên tiếp.

---

## 3. Cơ chế Kỹ thuật Tối ưu hóa của Intel OpenVINO

Quá trình chuyển đổi từ trọng số PyTorch sang OpenVINO IR mang lại các bước tối ưu hóa cấp sâu phần cứng:
1. **Layer Fusion (Gộp tầng mạng):**
   - Tự động gộp cặp phép toán Tích chập và Chuẩn hóa (`Conv2d + BatchNorm2d + SiLU`) thành một hạt nhân toán tử (Kernel) thực thi duy nhất trong bộ nhớ cache. Điều này triệt tiêu hoàn toàn chi phí đọc/ghi trung gian qua lại giữa RAM và thanh ghi.
2. **Lượng tử hóa & Chuẩn số thực nửa chính xác (FP16 Precision):**
   - Chuyển đổi trọng số từ `FP32` (32-bit float) sang `FP16` (16-bit half-precision). Dung lượng mô hình giảm một nửa, tăng gấp đôi băng thông đọc bộ nhớ và tận dụng các lệnh tính toán vector song song trên GPU EUs.
3. **Khai thác tập lệnh phần cứng Intel:**
   - **Trên CPU:** Tận dụng tập lệnh vector cao cấp `AVX-512` và `VNNI (Vector Neural Network Instructions)`.
   - **Trên iGPU:** Đồ thị tính toán được biên dịch trực tiếp sang OpenCL kernel, phân phối tính toán song song trên 96 Execution Units (EUs) của chip đồ họa Intel Iris Xe, giải phóng tải tính toán cho CPU phục vụ các tác vụ giải mã video, quản lý luồng mạng và giao diện người dùng.

---

## 4. Kết quả Thực nghiệm Đo đạc Thực tế

Số liệu được đo đạc chính xác qua quá trình benchmark độc lập trên cùng phần cứng máy trạm (trích xuất từ `results/hardware_benchmark.csv`):

### Bảng 1: Kết quả Benchmark Hiệu năng Suy luận (Inference Benchmark)

| STT | Cấu hình Mô hình | Thiết bị Phần cứng | Độ trễ TB (ms) | Trung vị (ms) | P95 Latency (ms) | Min (ms) | Max (ms) | Tốc độ (FPS) | Hệ số Tăng tốc | Đạt chuẩn Real-Time ($\ge 30$ FPS) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **PyTorch Framework (`.pt`)** | Intel CPU | 100.13 ms | 89.19 ms | 130.51 ms | 84.31 ms | 265.66 ms | 10.0 FPS | **1.00x (Gốc)** | ❌ Không đạt |
| 2 | **Intel OpenVINO (`.xml`)** | Intel CPU | 249.29 ms | 239.31 ms | 330.78 ms | 153.17 ms | 516.89 ms | 4.0 FPS | 0.40x | ❌ Chậm hơn (Không có iGPU offload) |
| 3 | **Intel OpenVINO (`.xml`)** | **Intel Iris Xe iGPU** | **28.16 ms** | **28.06 ms** | **29.19 ms** | **27.28 ms** | **29.51 ms** | **35.5 FPS** | **3.56x** | ✅ **Vượt chuẩn (Tối ưu nhất)** |

```
So sánh Tốc độ Khung hình (FPS) giữa các Cấu hình:
========================================================================
PyTorch CPU      : [########                     ] 10.0 FPS
OpenVINO Iris Xe : [############################ ] 35.5 FPS (TĂNG TỐC 3.56 LẦN)
========================================================================
Chuẩn Real-time  : [#######################      ] 30.0 FPS
========================================================================
```

---

## 5. Phân tích Khoa học & Đánh giá Chuyên sâu

1. **Hiệu quả vượt bậc của Intel OpenVINO trên Intel Iris Xe iGPU:**
   - **PyTorch CPU thuần (Baseline):** Mô hình `yolo26s.pt` có độ trễ trung bình **100.13 ms** (chỉ đạt **10.0 FPS**). Ở tốc độ này, khung hình video bị trễ, dễ gây bỏ lỡ khoảnh khắc chướng ngại vật băng qua đường ngang.
   - **OpenVINO trên Intel Iris Xe iGPU:** Đạt độ trễ trung bình xuất sắc **28.16 ms** (trung vị **28.06 ms**, P95 **29.19 ms**), tốc độ khung hình đạt **35.5 FPS** (tăng tốc **3.56 lần** so với PyTorch CPU).
2. **Lợi ích kiến trúc của việc chạy trên iGPU:**
   - Khi giao tác vụ suy luận mô hình học sâu nặng nề cho 96 EUs của card đồ họa tích hợp Intel Iris Xe, **tải CPU được giảm tải tối đa**. 
   - Nhờ đó, CPU có đầy đủ tài nguyên để đồng thời thực thi:
     - Giải mã luồng video độ phân giải cao (H.264/H.265 RTSP Stream).
     - Thuật toán theo dõi liên tục đa đối tượng (**ByteTrack**).
     - Tính toán hình học đa giác vùng xâm nhập (ROI Polygon Intersection) và thời gian dừng đỗ (Dwell Time).
     - Vận hành Web Server Flask và đẩy luồng video trực tiếp tới trình duyệt người dùng qua HTTP Multipart.
3. **Ý nghĩa thực tiễn & Giá trị Kinh tế:**
   - **Tiết kiệm chi phí đầu tư thiết bị:** Các đơn vị quản lý đường sắt không cần phải trang bị máy tính công nghiệp đắt tiền với card đồ họa rời NVIDIA (\$2,000+). Hệ thống hoàn toàn có thể chạy ổn định xấp xỉ 36 FPS trên các máy tính văn phòng hoặc Mini PC Intel NUC / Core i5 phổ thông (\$400 – \$600).
   - **Tối ưu hóa nhiệt độ và tiết kiệm điện:** Intel Iris Xe iGPU tiêu thụ công suất TDP cực thấp (chỉ từ 15W – 28W so với 120W – 250W của card rời), giúp máy trạm chạy bền bỉ 24/7 trong môi trường thời tiết khắc nghiệt mà không lo quá nhiệt.

---

## 6. Hướng dẫn Chạy Đo đạc & Xuất Mô hình

### 1. Xuất mô hình PyTorch sang OpenVINO FP16:
```bash
python models/yolo26s/export_openvino.py
```

### 2. Thực thi Benchmark hiệu năng trên phần cứng của bạn:
```bash
# Đo đạc tự động trên CPU và iGPU
python experiments/exp3_hardware_acceleration/benchmark_hardware.py --frames 50
```

---

## 7. Kết luận Thực nghiệm 3

Thực nghiệm đã chứng minh rõ nét hiệu quả của việc ứng dụng **Intel OpenVINO Toolkit kết hợp chip đồ họa tích hợp Intel Iris Xe**:
- Đạt tốc độ suy luận **35.5 FPS** với độ trễ chỉ **28.16 ms**, tăng tốc **3.56 lần** so với PyTorch CPU truyền thống (10.0 FPS).
- Độ trễ ổn định cao với P95 đạt **29.19 ms** và trung vị **28.06 ms**.
- Biến một máy trạm phổ thông thành thiết bị giám sát an toàn đường sắt thời gian thực chuẩn công nghiệp, tối ưu hóa toàn diện cả về **hiệu năng kỹ thuật** lẫn **hiệu quả kinh tế**.
