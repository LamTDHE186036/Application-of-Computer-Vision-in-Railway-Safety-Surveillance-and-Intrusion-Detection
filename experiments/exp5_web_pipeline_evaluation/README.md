# Thực nghiệm 5: Đánh giá Hiệu năng Thời gian chạy Luồng Web và Phân tích Ca Báo sót

Thư mục này chứa mã nguồn đo đạc và kết quả phục vụ **Mục 4.3.4** và **Mục 4.3.5** của báo cáo đồ án **RailSafe-CV v1.3.0**.

## 1. Cấu trúc tệp

- `evaluate_web_pipeline.py`: Script đo trực tiếp thời gian chạy từng công đoạn của luồng xử lý ứng dụng Web (`cap.read()` → `model.track()` → `process_and_draw()` → `cv2.imencode('.jpg', quality=80)`) và tự động thống kê 3 nhóm nguyên nhân báo sót / lọc bỏ trên ROI (Detector mất hộp bao, Tracker đổi ID, Thời gian trong ROI chưa đạt ngưỡng).
- `evaluate_events_offline.py`: Công cụ cài đặt chuẩn giao thức đánh giá ngoại tuyến ở cấp độ sự kiện được đề xuất ở Mục 4.3.4 (`tIoU`, `Event Precision`, `Event Recall`, `Event F1`, `FAR`, `Alert Detection Delay`), sẵn sàng sử dụng ngay khi có tệp nhãn sự kiện thực tế (`ground truth`).
- `results/web_pipeline_runtime.csv`: Bảng phân rã thời gian xử lý từng công đoạn đo trên video đường ngang thực tế (`demo1`, $1272 \times 720$, $30\ \mathrm{FPS}$, $774$ khung hình đo sau $20$ khung khởi động).
- `results/web_pipeline_diagnostics.json`: Nhật ký chi tiết các khoảng mất hộp bao, thống kê đổi ID và phân nhóm thời lượng của toàn bộ quỹ đạo đi vào vùng ROI.

## 2. Cách chạy tái lập

Chạy từ thư mục gốc dự án (`RailSafe-CV/`):

```bash
# Đo hiệu năng luồng Web và chẩn đoán ca báo sót trên video demo1
python experiments/exp5_web_pipeline_evaluation/evaluate_web_pipeline.py

# Hoặc chỉ định đường dẫn video kiểm thử bất kỳ
python experiments/exp5_web_pipeline_evaluation/evaluate_web_pipeline.py --video path/to/video.mp4

# Kiểm tra nhanh bộ chấm điểm cấp độ sự kiện (tIoU, Event P/R/F1, FAR, Delay)
python experiments/exp5_web_pipeline_evaluation/evaluate_events_offline.py --self-test
```

## 3. Tóm tắt kết quả đo thực tế trên `demo1` (YOLO26s OpenVINO FP16 trên Intel Iris Xe iGPU)

| Công đoạn trên luồng Web | Trung bình (ms) | Trung vị (ms) | P95 (ms) | Tỷ trọng |
| :--- | :---: | :---: | :---: | :---: |
| 1. Đọc khung hình (`cap.read`) | 2.12 | 1.95 | 2.92 | 5.4% |
| 2. Suy luận + Tracking (`model.track`) | 31.47 | 30.99 | 34.84 | 79.7% |
| 3. Xử lý ROI + Cảnh báo + Vẽ phủ (`process_and_draw`) | 3.27 | 3.11 | 4.30 | 8.3% |
| 4. Mã hóa MJPEG (`JPEG quality=80`) | 2.60 | 2.48 | 3.35 | 6.6% |
| **Tổng toàn luồng Web** | **39.47** | **38.79** | **43.79** | **100.0% (25.3 FPS)** |

- **Detector mất hộp bao trong ROI:** 13 khoảng ngắn (1–7 khung hình), dài nhất là `Motorcycle ID:38` (7 khung liên tiếp, 0.23 s) do chói đèn pha & nhòe chuyển động và `Motorcycle ID:6` (6 khung, 0.20 s) ở rìa xa ROI.
- **Tracker đổi ID trong ROI:** 0 ca trên đoạn thử nghiệm nhờ bộ đệm ByteTrack (`track_buffer = 30`) nối liền các khoảng mất box ngắn.
- **Chưa đủ thời gian trong ROI:** 2 quỹ đạo `< 0.5 s` (`Person ID:26` 0.03 s, `Person ID:4` 0.00 s) không kích hoạt báo động đỏ; 1 quỹ đạo `0.5–1.0 s` (`Person ID:10` 0.83 s) bị hủy khi chốt sự kiện theo ngưỡng `MIN_EVENT_DURATION = 1.0 s`.
