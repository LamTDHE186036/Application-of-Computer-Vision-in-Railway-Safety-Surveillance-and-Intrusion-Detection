# Fine-tune YOLO26m chuẩn 4 lớp

`train_yolo4_chuan.py` là bản mã nguồn lưu cùng lượt huấn luyện ngày 13–14/11/2026. Lượt đó tạo ra checkpoint [yolo26m-4class-standard.pt](../../../models/minh_yolo26m_4class_standard/yolo26m-4class-standard.pt), chọn theo validation ở epoch 48 trong lịch 50 epoch.

- SHA-256 của mã nguồn `b44b358e9c21d6a4db4bc12f5b1e02430049c896dd82766da5a3490412062e95`.
- SHA-256 của checkpoint bàn giao `0230f0348c4fa89324e9fda9992ca13e095c9880b9798aa98ed35e2651c96701`.
- SHA-256 của `yolo26m.pt` COCO dùng để khởi tạo `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`.

## Chạy lại bằng Ultralytics

Lượt gốc dùng Python 3.11, Ultralytics 8.4.161 và PyTorch 2.7.1 với CUDA 12.8. Chuẩn bị `yolo26m.pt` COCO và giải nén bộ dữ liệu YOLO bốn lớp đã bàn giao. `data.yaml` nằm ở gốc thư mục dữ liệu, có 1.600 ảnh train, 200 ảnh validation và 200 ảnh test. Thứ tự lớp là `person`, `two_wheeler`, `car`, `train`.

Thay hai đường dẫn đầu vào trong lệnh sau bằng đường dẫn trên máy của mình.

```sh
yolo detect train model="weights/yolo26m.pt" data="DATA-MINH-YOLO4-2004/data.yaml" epochs=50 patience=20 imgsz=640 batch=0.5 device=0 workers=0 optimizer=auto seed=20261113 deterministic=True amp=True cache=False save_period=1 project=runs/minh_yolo26m_4class_standard name=train
```

`batch=0.5` cho AutoBatch chọn batch theo GPU. GPU RTX 3050 Laptop 4 GB đã chọn batch 1. `optimizer=auto` đã chọn AdamW với learning rate khởi đầu 0,00125 trong lượt gốc. Có thể đặt `batch=1` để cố định batch đã ghi nhận. Train dùng `freeze=None` và các phép biến đổi ảnh mặc định của Ultralytics 8.4.161. Checkpoint của lượt mới nằm trong `weights/best.pt` bên trong thư mục kết quả mà Ultralytics in ra.

## Kiểm tra mã nguồn đã lưu

Script Python giữ cấu trúc workspace gốc để đối chiếu với lượt huấn luyện. Nó dùng `data/curated/railway-yolo4-full-2004-20261113-v1/`, kiểm hash từng cặp ảnh/nhãn, nạp `nghien-cuu/fine-tune/models/yolo26m.pt` và chạy bằng môi trường `ung-dung/.venv/` từ workspace đó. Với repo này hoặc ZIP dữ liệu đã bàn giao, dùng lệnh `yolo detect train` ở trên. Trong workspace gốc, `--check-only` kiểm đầu vào và GPU, còn chạy script mặc định sẽ bắt đầu huấn luyện.
