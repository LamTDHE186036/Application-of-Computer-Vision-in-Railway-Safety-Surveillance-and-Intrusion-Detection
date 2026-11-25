# Fine-tune YOLO26m trên Dataset_Lam

Đây là bộ dữ liệu Minh đã chọn làm mốc chính sau lượt fine-tune ngày 24/11/2026. `model/best.pt` là checkpoint epoch 13 đang dùng trên web demo. Lượt huấn luyện dừng sau 19 epoch và đạt mAP50–95 nội bộ 82,32% tại epoch 13 trên 532 ảnh validation.

| Đường dẫn | Nội dung |
|---|---|
| `model/best.pt` | Checkpoint fine-tune dùng suy luận. SHA-256 `5146bf56882d88a9216517c03ee07a238aba06f59ff4fbea8d90917653967bfb` |
| `Fine-tune-YOLO26m/model/yolo26m-coco.pt` | Trọng số COCO dùng để khởi tạo khi chạy lại fine-tune |
| `nghien-cuu/fine-tune/finetune_dataset_lam.py` | Script gốc của lượt train, giữ nguyên cấu hình và thứ tự lớp |
| `data/Dataset_Lam/Dataset/Dataset_all/` | 2.125 ảnh train nguồn, 532 ảnh validation và nhãn YOLO đi kèm. Script giữ riêng 25 ảnh train sau đối chiếu, dùng 2.100 ảnh train |
| `data/Dataset_Lam/Dataset_test/` | 404 ảnh đối chứng và một video mẫu. Các bộ đối chứng có YAML riêng |
| `training-record/` | Kết quả theo epoch, cấu hình lượt train và danh sách ảnh/nhãn được chọn kèm SHA-256 |

Thứ tự lớp là `0 car`, `1 motorcycle`, `2 person`, `3 truck-bus`, `4 train`. Validation có thể chia sẻ cảnh với train. Ứng dụng chỉ dùng `best.pt` để suy luận. Lệnh train bên dưới bắt đầu một lượt mới từ trọng số COCO.

## Chạy trên Windows

Trong Đồ án, file `.cmd` dùng môi trường Python 3.11 có sẵn của dự án. Nếu chuyển cả thư mục sang máy khác, chuẩn bị `.venv` với Ultralytics `8.4.161`, PyTorch `2.7.1+cu128` và torchvision `0.22.1+cu128` tương thích. Lượt train cần GPU NVIDIA và PyTorch nhận CUDA.

```powershell
.\Train-Dataset-Lam.cmd --check-only
.\Train-Dataset-Lam.cmd --train
```

Lệnh đầu kiểm dữ liệu, CUDA, loss và gradient với 0 bước optimizer. Lệnh thứ hai ghi một lượt mới vào `ket-qua/runs/` bên trong thư mục này.
