# Đánh giá YOLO26m trên bộ test 119 ảnh

Phép đánh giá trong thesis so sánh YOLO26m gốc COCO và YOLO26m fine-tune epoch 13 trên 119 ảnh với 707 hộp nhãn. Cả hai checkpoint được chấm theo năm lớp `car`, `motorcycle`, `person`, `truck-bus` và `train`. Baseline dùng điểm lớn hơn giữa hai lớp COCO `truck` và `bus` cho lớp `truck-bus`.

## Chuẩn bị

- Dùng Python 3.11 với Ultralytics 8.4.161. Máy đánh giá dùng PyTorch 2.7.1 + CUDA 12.8. Môi trường cần có OpenCV, NumPy, PyYAML và Matplotlib.
- Trỏ tới thư mục `dataset_test` chứa `roboflow_crossing/data.yaml`, `roboflow_crossing/images/` và `roboflow_crossing/labels/`. Script đọc thứ tự lớp từ `data.yaml`.
- Dùng hai checkpoint của phép đánh giá, `yolo26m-coco.pt` và `best.pt` epoch 13. Script kiểm SHA-256 của cả hai checkpoint trước khi suy luận.

## Chạy từ thư mục gốc repo

```text
python experiments/exp4_yolo26m_thesis/score.py path/to/dataset_test path/to/yolo26m-coco.pt path/to/best.pt ../runs/yolo26m-thesis-119 --source roboflow_crossing --device 0
python experiments/exp4_yolo26m_thesis/plot_confusion_matrix.py ../runs/yolo26m-thesis-119
```

Thay các đường dẫn `path/to/...` bằng đường dẫn trên máy. Chọn một thư mục kết quả mới cho mỗi lượt chấm. Dùng `--device cpu` nếu máy chạy bằng CPU.

## Đọc kết quả

- `metrics.json` lưu kết quả theo nguồn và theo lớp, gồm Precision, Recall, F1, AP50 và AP50–95. Các chỉ số ở ngưỡng cố định nằm trong `by_source.roboflow_crossing.fixed`. Đọc `micro` để lấy Precision và Recall tổng tại confidence 0,35 và IoU 0,50.
- `manifest.json` ghi danh sách và SHA-256 của ảnh, nhãn. `protocol.json` ghi cấu hình, phiên bản thư viện và SHA-256 của checkpoint.
- `confusion-matrix-test-119.png` là hình so sánh và `confusion-matrices.json` lưu số đếm. Script giữ các cặp đúng lớp từ phép chấm ở ngưỡng cố định, sau đó ghép một-một các hộp khác lớp còn lại theo thứ tự IoU giảm dần. Hàng là nhãn thật và cột là dự đoán. Hàng Background đếm dự đoán chưa ghép, cột Background đếm hộp nhãn chưa ghép.

| Mô hình | mAP50 | mAP50–95 | Precision (micro) | Recall (micro) |
| --- | ---: | ---: | ---: | ---: |
| YOLO26m gốc COCO | 82,46% | 64,51% | 83,67% | 65,21% |
| YOLO26m fine-tune epoch 13 | 86,11% | 65,54% | 79,57% | 73,27% |
