# Train YOLO26m

Dùng Ultralytics 8.4.161, trọng số YOLO26m gốc và bộ dữ liệu 4 lớp. Sửa hai đường dẫn đầu vào theo vị trí file trên máy.

```sh
yolo detect train model="weights/yolo26m.pt" data="dataset/data.yaml" epochs=50 patience=20 imgsz=640 batch=1 workers=0 optimizer=auto seed=20261113
```

[Mã nguồn](train_yolo4_chuan.py) · [Model đã train](../../../models/yolo26m/README.md)
