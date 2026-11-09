# YOLO26m fine-tune ảnh thật và ảnh tổng hợp

Đóng góp của Minh cho thư mục nghiên cứu fine-tune. Checkpoint YOLO26m đã học được lưu tại `models/yolo26m-that-ai.pt` (44.053.303 byte).

SHA-256 `abe022e9d2b27f39f45507cdd289e7e152853a0255d776a3ad1b9c9aeed051de`.

## Nhóm đối tượng

| ID | Nhóm | Nội dung |
|---|---|---|
| 0 | person | Người, gồm người lái nếu nhìn rõ và có nhãn riêng |
| 1 | two_wheeler | Xe đạp và xe máy |
| 2 | large_vehicle | Xe tải, bus, pickup, van cao/minibus |
| 3 | car | Ô tô con, SUV, MPV/minivan gia đình |

Nhánh nghiên cứu này dùng đầu phân loại gộp `GroupedClassifier`. Khi load checkpoint cần giữ thư mục `railway/` bên cạnh script để Python tìm được lớp `railway.grouped_head.GroupedClassifier` đã lưu trong checkpoint.

## Chạy thử checkpoint

Dùng Python 3.11 và môi trường riêng cho thí nghiệm để cố định các phiên bản. Từ thư mục gốc repo, chạy PowerShell

```powershell
py -3.11 -m venv research/finetune/minh_yolo26m_that_ai/.venv
.\research\finetune\minh_yolo26m_that_ai\.venv\Scripts\python.exe -m pip install -r research/finetune/minh_yolo26m_that_ai/requirements.txt
.\research\finetune\minh_yolo26m_that_ai\.venv\Scripts\python.exe research/finetune/minh_yolo26m_that_ai/check_model.py
.\research\finetune\minh_yolo26m_that_ai\.venv\Scripts\python.exe research/finetune/minh_yolo26m_that_ai/check_model.py --image C:/data/example.jpg --device cpu
```

Chạy với ảnh trống chỉ kiểm tra khả năng load và suy luận. Đánh giá chất lượng cần ảnh thật có nhãn và giao thức dùng chung.

## Huấn luyện

`finetune_that_ai.py` chứa công thức gốc với 700 ảnh thật, 700 ảnh tổng hợp, 125 ảnh validation thật. Chỉ cập nhật 9.252 tham số phân loại, dùng nhãn từng nguồn để bỏ loss cho các nhóm chưa được gán nhãn. Cấu hình mặc định gồm 3 epoch, ảnh 640, batch 8, AdamW, learning rate 0,00005 và seed 20261001. Chọn checkpoint bằng validation.

Cần chuẩn bị thêm checkpoint COCO khởi tạo `models/yolo26m.pt` trong thư mục này và bộ dữ liệu có manifest tại `data/prepared/finetune-mixed-20261001` tính từ gốc repo. Bộ dữ liệu cần có cấu trúc và hash tương ứng manifest gốc. Có thể chỉnh `MODEL_GOC`, `DU_LIEU`, `NOI_LUU` ở đầu script cho máy đang dùng. Huấn luyện yêu cầu PyTorch CUDA và GPU NVIDIA. Cài bản PyTorch CUDA phù hợp trong môi trường riêng trước khi chạy train.

```powershell
.\research\finetune\minh_yolo26m_that_ai\.venv\Scripts\python.exe research/finetune/minh_yolo26m_that_ai/finetune_that_ai.py --check-only
.\research\finetune\minh_yolo26m_that_ai\.venv\Scripts\python.exe research/finetune/minh_yolo26m_that_ai/finetune_that_ai.py
```

Các lượt thử bổ sung vd1 và vùng bỏ qua được quản lý trong workspace nghiên cứu riêng. Thư mục này cung cấp công thức thật + AI của checkpoint đã chốt.

## Tích hợp với web

Web tại commit `71352d0` ưu tiên YOLO26s OpenVINO và có 5 lớp theo thứ tự `car, motorcycle, person, truck-bus, train`. Nhánh này có 4 lớp theo bảng trên. Khi tích hợp cần nạp module custom, đổi đường dẫn model và ánh xạ tên `two_wheeler` sang nhóm xe máy/xe đạp, `large_vehicle` sang nhóm xe lớn. Model này chưa có đầu ra `train`, nên chức năng lọc tàu của web cần được đánh giá riêng.

Tốc độ OpenVINO của YOLO26s và số liệu trong README chính thuộc model và thí nghiệm của Lâm. Đánh giá model đóng góp này cần dùng cùng dữ liệu, phiên bản nhãn và cấu hình với model đối chứng.
