# YOLO26m fine-tune 5 lớp

Gói suy luận độc lập gồm checkpoint đang triển khai và lớp Python cần để nạp checkpoint. Không thay model mặc định, giao diện, tracking hoặc cấu hình của ứng dụng.

## Sử dụng

Chạy từ thư mục gốc repo, với Python 3.11 và thư viện trong `requirements.txt` của gói. Đã kiểm tra với PyTorch 2.7.1 và Ultralytics 8.4.161. Gói này dùng backend PyTorch; không phải model OpenVINO.

```python
from models.minh_yolo26m_5class import load_model, to_lam_class_id, to_lam_label

model = load_model(device='cpu')  # device=0 cho CUDA
results = model.predict(frame, verbose=False)
for box in results[0].boxes:
    native_id = int(box.cls[0])
    lam_id = to_lam_class_id(native_id)
    lam_label = to_lam_label(model.names[native_id])
    xyxy = box.xyxy[0].tolist()
    confidence = float(box.conf[0])
```

`model.predict()` / `model.track()` trả kết quả YOLO chuẩn với **ID gốc của model này**. Các hàm chuyển nhãn chỉ chuyển ID/tên để bên tích hợp sử dụng; không sửa box, confidence hoặc đối tượng `Results`.

| ID model này | Tên gốc | ID model hiện có của repo | Tên dùng ở repo |
|---:|---|---:|---|
| 0 | person | 2 | person |
| 1 | two_wheeler | 1 | motorcycle |
| 2 | large_vehicle | 3 | truck-bus |
| 3 | car | 0 | car |
| 4 | train | 4 | train |

`two_wheeler` gồm xe đạp và xe máy. `car` gồm sedan, hatchback, SUV và MPV/minivan gia đình. `large_vehicle` gồm bus, truck, pickup, van chở hàng và minibus thân dài. `motorcycle` / `truck-bus` ở cột cuối chỉ là tên tương thích với repo.

Lớp tàu dùng để hiển thị/giám sát; logic có phát cảnh báo xâm nhập cho tàu hay không thuộc bên tích hợp. Gói không tự thêm quy tắc ROI hoặc cảnh báo.

## Checkpoint

Đây là bản fine-tune 5 lớp đã được chọn bằng validation, epoch 3. Khi đóng gói chỉ đổi đường dẫn lớp Python sang `models.minh_yolo26m_5class.layers`; toàn bộ tensor được kiểm tra bằng nhau với bản triển khai. Không kèm dữ liệu, video, báo cáo thực nghiệm hoặc model thử nghiệm.

`model.json` ghi SHA-256 và cấu hình suy luận. Giữ nguyên thư mục gói khi sử dụng: checkpoint có lớp Python tùy biến và không thể chỉ chép riêng `.pt` rồi bỏ `layers.py`.

Checkpoint này dành cho suy luận, không chứa optimizer để tiếp tục train. Phụ thuộc Ultralytics sử dụng giấy phép AGPL-3.0 / Enterprise theo [thông tin bản quyền Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
