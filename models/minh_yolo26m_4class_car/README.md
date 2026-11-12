# YOLO26m — 4 lớp, gộp phương tiện thành car

Checkpoint suy luận độc lập: **`yolo26m-4class-car.pt`**. Nạp trực tiếp bằng Ultralytics; không cần `layers.py`, `inference.py` hoặc module Python riêng của dự án nghiên cứu.

## Nhãn đầu ra

| ID | Tên trong model | Phạm vi |
|---:|---|---|
| 0 | `person` | Người |
| 1 | `two_wheeler` | Xe đạp và xe máy |
| 2 | `car` | Ô tô con và xe lớn, gồm bus, truck, pickup, van, minibus theo quy ước của model nguồn |
| 3 | `train` | Tàu hỏa; giữ riêng với phương tiện đường bộ |

Model không còn tách `car` và `truck-bus`/`large_vehicle`. ID này khác model YOLO26s hiện có của repo (`car=0`, `motorcycle=1`, `person=2`, `truck-bus=3`, `train=4`). Khi xử lý kết quả, dùng `results[0].names[class_id]` và danh sách bốn lớp trên; không giữ bảng ID năm lớp cũ.

## Nạp và nhận diện

Môi trường đã kiểm tra: Python 3.11, PyTorch 2.7.1, **Ultralytics 8.4.161**. Backend của file này là PyTorch, dùng CPU hoặc CUDA; chưa kèm model OpenVINO.

Chạy từ thư mục gốc repo, với `frame` là ảnh NumPy đọc từ video/camera:

```python
from ultralytics import YOLO

model = YOLO("models/minh_yolo26m_4class_car/yolo26m-4class-car.pt")
results = model.predict(
    frame, imgsz=640, conf=0.50, nms=False, device="cpu", verbose=False
)

for box in results[0].boxes:
    class_id = int(box.cls[0])
    name = results[0].names[class_id]
    xyxy = box.xyxy[0].tolist()
    confidence = float(box.conf[0])
```

Có thể dùng `model.track()` với ByteTrack, `persist=True` và `nms=False`; việc quản lý ID, ROI, thời gian dừng và cảnh báo vẫn thuộc ứng dụng tích hợp. `device=0` dùng CUDA khi có GPU NVIDIA phù hợp.

## Các điểm tích hợp vào web của repo

1. **Chọn đúng file model:** `app/services/detector.py::load_yolo_model()` hiện ưu tiên thư mục OpenVINO của YOLO26s. Thêm/chọn đường dẫn `.pt` ở trên khi thử model này; chỉ chép file vào `models/` không tự đổi model đang chạy.
2. **Cập nhật tên lớp:** dùng `person`, `two_wheeler`, `car`, `train`. Đổi tên tương ứng trong `CLASS_NAMES_VI`, `app/config.py::DEFAULT_CLASSES`, `configs/classes.json` và các lựa chọn lớp ở giao diện. `two_wheeler` thay tên tương thích `motorcycle`; `car` bao gồm cả nhóm `truck-bus` cũ.
3. **Giữ tàu ngoài cảnh báo xâm nhập:** kiểm tra theo `name == "train"` (ID mới là **3**, không phải 4). Tàu không đi vào bộ đếm dừng của người/xe hoặc sự kiện xâm nhập. Trạng thái tàu sắp tới/không có tàu vẫn do logic ứng dụng quyết định.
4. **Tracking:** cập nhật các bảng lớp/pool nếu có hard-code năm ID, đưa mọi xe thuộc `car` vào cùng nhóm, và reset tracker khi đổi model. Không suy ra xe con/xe lớn từ kết quả `car` vì phân biệt đó đã được gộp.
5. **Thiết bị:** đường dẫn PyTorch dùng `cpu` hoặc CUDA. Muốn dùng Intel GPU qua OpenVINO cần xuất và kiểm tra riêng, không chọn thư mục OpenVINO YOLO26s cũ để chạy trọng số mới này.

Gói này chỉ bổ sung checkpoint và hướng dẫn; không thay model mặc định, mã ứng dụng, cấu hình hoặc tracking của repo.

## Nguồn và kiểm tra

- Dẫn xuất từ YOLO26m fine-tune 5 lớp đã chọn bằng validation, epoch 3; đây không phải lượt train đầu phân loại chuẩn mới.
- Giữ nguyên toàn bộ **768 tensor nguồn**; thêm các phép chọn/gộp kênh cố định bằng lớp PyTorch có sẵn. Không có optimizer step mới.
- Gộp `large_vehicle` và `car` bằng **maximum logit trước top-k**; không cộng confidence, không thêm NMS. Các nhóm người, xe hai bánh và tàu được giữ riêng.
- Bản đóng gói đối chứng 5 lớp cho dự đoán trùng hoàn toàn model nguồn trên 455 ảnh và 900 frame; bản 4 lớp đã được chấm riêng sau gộp nhãn. Đây là kiểm tra nội bộ, chưa chứng minh chất lượng trên camera mới.
- File nạp và suy luận được trong Python cách ly không có module nghiên cứu hoặc hai file hỗ trợ cũ.
- Checkpoint dành cho **suy luận**, không chứa optimizer để resume. Không dùng trainer mặc định dựng lại kiến trúc từ YAML để tiếp tục train bản này; cần quy trình huấn luyện giữ đúng cấu trúc gộp và xử lý nhãn một phần/vùng bỏ qua.

SHA-256 của `yolo26m-4class-car.pt`:

```text
9d8d2ca82c0c8eb1ea60e0eaa7da15f31d968b51205b40aa9483b78c5bac9e9a
```

SHA-256 checkpoint nguồn 5 lớp:

```text
2b100b32afe7d37cf21770ce38ecc0db8340e1caafef92ce8afc3b4d1eac923c
```
