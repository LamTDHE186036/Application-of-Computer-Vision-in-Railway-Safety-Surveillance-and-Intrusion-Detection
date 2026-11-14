# YOLO26m fine-tune chuẩn 4 lớp

`yolo26m-4class-standard.pt` là checkpoint `best.pt` được chọn ở epoch 48 theo validation trong lượt train đủ 50 epoch ngày 13–14/11/2026. Model học trực tiếp bốn lớp từ YOLO26m pretrained COCO bằng Ultralytics 8.4.161.

| ID | Nhãn | Phạm vi |
|---:|---|---|
| 0 | `person` | Người |
| 1 | `two_wheeler` | Xe đạp, xe máy |
| 2 | `car` | Ô tô, xe buýt, xe tải, tuk-tuk |
| 3 | `train` | Tàu đường sắt, tàu điện/tram |

Nạp file `.pt` trực tiếp bằng Python 3.11, PyTorch 2.7.1 và Ultralytics 8.4.161 đã dùng khi kiểm tra. Chạy ví dụ từ thư mục gốc repo, với `frame` là ảnh đầu vào.

```python
from ultralytics import YOLO

model = YOLO("models/minh_yolo26m_4class_standard/yolo26m-4class-standard.pt")
results = model.predict(frame, imgsz=640, device="cpu", verbose=False)
```

Đọc `results[0].names` để lấy tên lớp. Khi thử tích hợp, giữ `train` ngoài logic cảnh báo xâm nhập. Repo cũng có bản [4 lớp gộp từ model cũ](../minh_yolo26m_4class_car/README.md) với tên checkpoint riêng.

| Tập | Ảnh | mAP50 | mAP50–95 |
|---|---:|---:|---:|
| Validation | 200 | 84,08% | 62,61% |
| Test nội bộ | 200 | 69,20% | 47,21% |

Tập train có 1.600 ảnh. Các điểm trên đo nhận diện, còn tracking và cảnh báo cần kiểm tra trên video. Trên cùng 200 ảnh test nội bộ, model web của Minh đạt 62,30% mAP50–95 sau khi quy về cùng bốn lớp. Trong đó có 142 ảnh từng dùng để train model web. Các tập ảnh dùng nhãn hiện có với mức rà soát khác nhau.

SHA-256 của `yolo26m-4class-standard.pt`:

```text
0230f0348c4fa89324e9fda9992ca13e095c9880b9798aa98ed35e2651c96701
```
