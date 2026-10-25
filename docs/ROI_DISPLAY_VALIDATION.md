# Kiểm thử hiển thị ROI — 25/10/2026

Đã chạy web Flask trên nhánh `feature/roi-only-box-display` với video thật
`03-foreign-day.mp4` (Glendale, camera cố định, 1920×1080, 30 FPS, 10 giây).
Model: `train-railway-yolo11s_openvino_model`, OpenVINO CPU, ảnh suy luận 640.
Thử nghiệm dùng confidence 0.50; đây là cấu hình thử, không thay cấu hình lưu trong repo.

## Thao tác và kết quả

- Tải video qua giao diện, vẽ/lưu ROI, bắt đầu và dừng giám sát.
- Theo dõi tổng cộng 623 khung hình suy luận qua hai chế độ; video có lặp lại.
  Đây không phải 623 ảnh độc lập để đánh giá độ chính xác.
- Đối chiếu đầu ra tracker thực với các lệnh vẽ box thực: 333 lượt đối tượng
  ngoài ROI đều bị ẩn, 318 lượt giao ROI đều được vẽ. Không có sai khác.
- Trong lượt đầu, xe giữ ID 4 ở ngoài, trong và sau khi rời ROI.
  Xe bắt đầu giao ROI tại 4.2667 giây, cảnh báo tại 4.7667 giây (chờ 0.5 giây).
  Box của ID 4 biến mất sau khi xe ra ngoài; tracker vẫn trả ID 4.
- Chế độ `no_train`: đặt thời gian 2 giây; lần vào ROI tại 4.2667 giây,
  cảnh báo tại 6.2667 giây. Box vàng trước ngưỡng, cảnh báo sau ngưỡng.
- Giao diện hiển thị cảnh báo xâm nhập, lịch sử tổng cộng 4 sự kiện và 2 video.
  Mở video sự kiện ID 4 từ lịch sử: trình duyệt phát tới hết video 8.2667 giây,
  không có lỗi phần tử video. Không có lỗi JavaScript trong lượt quan sát.
- `python -m unittest discover -s tests -v`: 5 kiểm tra logic đạt.

## Phạm vi

Chỉ thay điều kiện vẽ box/ID. Giữ nguyên quy tắc giao ROI, ngưỡng cảnh báo,
model và tracking toàn khung hình. Chưa thêm tracking giới hạn trong vùng đệm.
Không khẳng định chất lượng ID trên mọi cảnh, độ chính xác model hay FPS thời gian thực.
Chưa thử webcam vật lý, âm thanh cảnh báo hoặc máy của Lâm.

Bản gốc có chữ tiếng Việt trên box hiển thị dấu hỏi do cách vẽ chữ bằng OpenCV;
lượt sửa này không thay phần chữ. Thử nghiệm sử dụng một trình chạy quan sát cục bộ
để ghi đầu ra tracker/lệnh vẽ, không thay kết quả suy luận và không đưa trình chạy đó
vào mã ứng dụng.
