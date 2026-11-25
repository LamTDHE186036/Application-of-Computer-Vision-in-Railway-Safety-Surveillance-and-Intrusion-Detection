# THỰC NGHIỆM 2: ĐÁNH GIÁ & SO SÁNH GIẢI THUẬT TRACKING ĐA ĐỐI TƯỢNG (YOLO26s & YOLO26m)

## 1. Đặt vấn đề & Bối cảnh Thực tế trong An toàn Đường sắt

Sau **Thực nghiệm 1**, hai mô hình nòng cốt của chiến lược thích ứng phần cứng là **`YOLO26s`** (Edge Real-Time) và **`YOLO26m`** (High-Accuracy) được đưa vào đánh giá cùng các giải thuật **Theo dõi đa đối tượng (Multi-Object Tracking — MOT)**.

Nếu chỉ sử dụng phát hiện đơn khung hình (**Pure YOLO**), hệ thống không có định danh liên tục (`Track ID`) nên không thể đo thời gian lưu trú (*Dwell Time*, mặc định `5.0s`, tùy chỉnh `1–60s` ở chế độ *Không có tàu*) và dễ phát báo động giả khi phương tiện băng qua đường ngang hợp pháp hoặc bị che khuất tạm thời.

Thực nghiệm so sánh trực tiếp 3 phương pháp trên **cả `YOLO26s` lẫn `YOLO26m`**:
1. **Pure YOLO (No Tracking):** Baseline chỉ chạy detector từng khung hình.
2. **BoT-SORT:** Theo dõi kết hợp Kalman Filter + bù chuyển động camera **GMC (`sparseOptFlow`)**.
3. **ByteTrack (Đề xuất):** Liên kết hộp bao **2 giai đoạn (High-score & Low-score IoU)** + Kalman Filter (cấu hình `bytetrack.yaml`: `track_high_thresh=0.25`, `track_low_thresh=0.1`, `track_buffer=30`, `match_thresh=0.8`, `fuse_score=True`).

---

## 2. Thiết lập Đo đạc

- **Video thực địa:** `dataset_test/test_exp.mp4` (`150` khung hình đầu nạp trực tiếp vào RAM để loại trừ độ trễ đọc đĩa, `imgsz=640`, `conf=0.25`).
- **Backend thực thi:** `Intel OpenVINO FP16` trên **Intel Iris Xe Graphics (`intel:gpu`)** — phản ánh chính xác pipeline vận hành thực tế với độ ổn định xung nhịp cao (giảm thiểu nhiễu thermal throttling của CPU).
- **Quy trình đo công bằng:** Mỗi mô hình được nạp **1 instance duy nhất**, chạy `20` khung hình warmup ổn định xung nhịp phần cứng, sau đó reset tracker sạch và chạy `10` khung hình warmup riêng trước từng chế độ (`Pure YOLO`, `BoT-SORT`, `ByteTrack`).

---

## 3. Kết quả Đo đạc Thực tế
*(Trích xuất trực tiếp từ `results/tracker_comparison.csv`)*

| Mô hình Detector | Phương pháp Tracking | Độ trễ TB (ms) | P95 Latency (ms) | Overhead so với Pure YOLO | Tốc độ (FPS) | Số ID độc lập | Vòng đời vệt TB | Số lần đứt ID | Hỗ trợ Dwell Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`YOLO26s`** | Pure YOLO (No Tracking) | 29.27 ms | 32.46 ms | 0.00 ms (Gốc) | 34.2 FPS | N/A | 0.0 frames | N/A | ❌ Không thể đo |
| *(OpenVINO iGPU)* | BoT-SORT | 33.94 ms | 35.65 ms | +4.67 ms | 29.5 FPS | 2 | 126.0 frames/ID | 6 | ⚠️ Tốn thêm GMC optical flow |
| | **ByteTrack (Đề xuất)** | **29.54 ms** | **30.64 ms** | **+0.27 ms** | **33.9 FPS** | **2** | **126.0 frames/ID** | **6** | ✅ **Tối ưu nhất (-94.2% overhead)** |
| **`YOLO26m`** | Pure YOLO (No Tracking) | 75.63 ms | 86.35 ms | 0.00 ms (Gốc) | 13.2 FPS | N/A | 0.0 frames | N/A | ❌ Không thể đo |
| *(OpenVINO iGPU)* | BoT-SORT | 87.06 ms | 117.26 ms | +11.43 ms | 11.5 FPS | 2 | 129.5 frames/ID | 7 | ⚠️ Overhead & P95 tăng mạnh |
| | **ByteTrack (Đề xuất)** | **75.17 ms** | **81.30 ms** | **≈ 0 ms** *(-0.46 ms)* | **13.3 FPS** | **2** | **129.5 frames/ID** | **7** | ✅ **Tối ưu nhất (nhanh hơn 11.89 ms)** |

*(Ghi chú: Giá trị `-0.46 ms` trên `YOLO26m + ByteTrack` cho thấy chi phí thuật toán ByteTrack siêu nhẹ `< 0.5 ms/frame`, nằm gọn trong biên độ dao động xung nhịp tự nhiên của iGPU).*

---

## 4. Phân tích Khoa học & Thảo luận

1. **Chi phí bổ sung (Tracking Overhead) gần như bằng 0 với ByteTrack:**
   - Trên **`YOLO26s`**, **ByteTrack** chỉ cộng thêm **`+0.27 ms/frame`** (giữ vững **`33.9 FPS`** và P95 **`30.64 ms`**), trong khi **BoT-SORT** tốn thêm **`+4.67 ms/frame`** (kéo FPS xuống dưới ngưỡng 30 FPS còn `29.5 FPS`) do phải chạy bù chuyển động camera `sparseOptFlow` không cần thiết cho camera đường sắt lắp cố định.
   - Trên **`YOLO26m`**, **ByteTrack** nhanh hơn **BoT-SORT** tới **`11.89 ms/frame`** (`75.17 ms` vs `87.06 ms`) và giảm mạnh độ trễ đuôi P95 từ `117.26 ms` xuống **`81.30 ms`** (`-35.96 ms`).
2. **Độ ổn định định danh (ID Consistency) tương đương mà không cần Re-ID / GMC:**
   - Cả BoT-SORT và ByteTrack đều đạt chất lượng bám vết giống hệt nhau trên cùng detector (`126.0 frames/ID` với `YOLO26s` và `129.5 frames/ID` với `YOLO26m`), nhưng ByteTrack không tốn chi phí tính toán optical flow hay trích xuất đặc trưng ngoại hình.
   - Ở camera đường sắt cố định (static CCTV), ByteTrack tận dụng liên kết IoU 2 tầng kết hợp `track_buffer = 30 frames` để duy trì ID ổn định qua che khuất tạm thời và phục vụ bộ đếm *Dwell Time* chính xác.

---

## 5. Hướng dẫn Tái hiện Thực nghiệm

```bash
# Chạy đánh giá Tracking cho cả YOLO26s và YOLO26m trên OpenVINO iGPU (mặc định auto)
python experiments/exp2_tracker_comparison/benchmark_tracker.py --frames 150 --backend openvino

# Hoặc chạy trên PyTorch CPU
python experiments/exp2_tracker_comparison/benchmark_tracker.py --frames 150 --backend pytorch
```

Kết quả CSV và biểu đồ trực quan được lưu tại `results/tracker_comparison.csv` và `results/chart_tracker_comparison.png`.

---

## 6. Kết luận Thực nghiệm 2

**ByteTrack** chứng minh tính ưu việt tuyệt đối trên **cả `YOLO26s` lẫn `YOLO26m`**:
- Overhead siêu nhẹ (`+0.27 ms` trên `YOLO26s`, giảm **`94.2%`** so với BoT-SORT; nhanh hơn BoT-SORT **`11.89 ms`** trên `YOLO26m`).
- Giữ vững **`33.9 FPS`** trên `YOLO26s + OpenVINO iGPU`, đảm bảo độ trễ P95 mượt mà và duy trì Track ID bền bỉ cho bài toán đo *Dwell Time*.
- **ByteTrack chính thức được chọn làm bộ theo dõi mặc định cho toàn bộ hệ thống RailSafe-CV v1.3.0.**
