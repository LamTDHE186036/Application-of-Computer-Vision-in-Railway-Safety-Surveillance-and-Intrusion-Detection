# Quy trình fine-tune và xuất model 4 lớp gộp car

Code của quy trình tạo ra [`yolo26m-4class-car.pt`](../../../models/minh_yolo26m_4class_car/yolo26m-4class-car.pt). Gồm hai giai đoạn học và một giai đoạn gộp/xuất. Bản 4 lớp cuối không được huấn luyện lại từ đầu.

## Các file chính

| File | Chức năng |
|---|---|
| `finetune_that_ai.py` | Fine-tune 4 lớp gốc: `person`, `two_wheeler`, `large_vehicle`, `car`; chọn 700 ảnh thật + 700 ảnh tổng hợp, 125 ảnh validation thật. |
| `finetune_5_lop.py` | Học tiếp checkpoint 4 lớp gốc, thêm `train`; dùng 1.779 ảnh train và 175 ảnh validation. |
| `railway_five_class.py` | Cấu trúc 5 lớp, loader, loss và validation có xử lý coverage/vùng bỏ qua. |
| `export_portable_car.py` | Giữ trọng số 5 lớp, gộp `large_vehicle` + `car`, xuất checkpoint chỉ dùng lớp PyTorch có sẵn; không có optimizer step. |
| `prepare_car_four_class.py` | Chuyển nhãn/mask sang 4 lớp mới trong thư mục dữ liệu riêng. |
| `railway/` | Định nghĩa tương thích để đọc checkpoint nghiên cứu/triển khai cũ. Chỉ cần cho quy trình nghiên cứu này. |
| `spatial_ignore.py`, `verify_spatial_ignore.py` | Adapter và kiểm tra tùy chọn cho dữ liệu vd1 có mask; không thêm dữ liệu này vào công thức gốc nếu không chỉ định `--ignore-data`. |
| `training_utils.py` | Kiểm tra hash, ghi metadata, tương thích checkpoint đã công bố. |

**Phân biệt hai bộ 4 lớp:** giai đoạn đầu có `large_vehicle` và chưa có tàu; bản bàn giao cuối có `train` và đã gộp xe lớn vào `car`.

## Công thức được giữ nguyên

| Giai đoạn | Lớp | Tham số được học | Epoch tối đa | Batch / NBS | AdamW LR | AMP |
|---|---|---:|---:|---|---:|---|
| Thật + tổng hợp | 4 lớp gốc | 9.252 tham số phân loại | 3 | 8 / 64 | 0,00005 | BF16 |
| Thêm tàu | 5 lớp | 10.794 tham số phân loại | 10, patience 4 | 8 / 64 | 0,00001 | Tắt |
| Gộp car / xuất | 4 lớp mới | Không cập nhật trọng số | 0 | — | — | — |

Backbone, neck, nhánh box và BatchNorm được giữ cố định khi fine-tune. Các phép biến đổi ảnh như mosaic, lật, đổi màu, dịch/co giãn bị tắt trong công thức này. Best checkpoint được chọn bằng validation, không chọn theo test.

Trong lượt đã tạo model nguồn, giai đoạn đầu chạy 3 epoch; giai đoạn thêm tàu chạy 7 epoch và chọn epoch 3. Chạy lại trên phần cứng/phiên bản khác không bảo đảm tái tạo từng byte hoặc chọn cùng epoch.

Phần thích nghi so với code tại workspace: đầu vào/đầu ra có tham số dòng lệnh, không ép dùng môi trường Python riêng của workspace cũ, nhận cả bộ dữ liệu gốc và bản 1.400 ảnh đã chọn trong gói bàn giao. Công thức học, loss, phép gộp và bảo toàn trọng số không đổi. `source-provenance.json` ghi hash các file nguồn trước khi thích nghi.

## Môi trường và đầu vào

Python 3.11 và các thư viện trong `requirements.txt`; Ultralytics **8.4.161**, PyTorch **2.7.1**. Cài PyTorch CUDA phù hợp GPU NVIDIA trước khi train. `--check-only` không chạy optimizer và không bắt đầu epoch huấn luyện.

Dùng môi trường Python riêng cho phần nghiên cứu, có thể đặt tại `.venv/` trong thư mục này. Các script dùng môi trường đó nếu có; nếu không, dùng Python đang gọi lệnh.

Giai đoạn đầu giữ BF16 như lượt gốc và cần GPU hỗ trợ chế độ đó. Nếu đổi sang FP16/FP32 trên máy khác, ghi nhận thay đổi cấu hình và kiểm tra validation lại.

Không kèm dataset hoặc các checkpoint trung gian trong thư mục code này. Cần:

- YOLO26m COCO khởi tạo `yolo26m.pt`.
- Dataset giai đoạn 4 lớp và giai đoạn 5 lớp có manifest/nhãn tương ứng. Trong ZIP dữ liệu đã bàn giao, hai thư mục là `01_finetune_4class_real_ai` và `02_finetune_5class`.
- Nếu dùng ZIP đó, chạy `prepare_paths.py` đi kèm dữ liệu sau khi giải nén để manifest, YAML, contract và mask trỏ đúng vị trí trên máy mới. Không chỉ sửa đường dẫn `data.yaml`: loader còn kiểm tra đường dẫn và hash trong contract.
- Để bắt đầu trực tiếp từ giai đoạn thêm tàu, cần checkpoint **4 lớp gốc** có grouped classifier. Model 4 lớp gộp car đã công bố không phải đầu vào của giai đoạn này.
- Để chỉ gộp/xuất, cần checkpoint **5 lớp nguồn** còn grouped classifier. Có thể dùng checkpoint nghiên cứu vừa train hoặc bản 5 lớp đã công bố trước; không dùng bản portable đã gộp để gộp lần nữa.

## 1. Fine-tune 4 lớp gốc

Ví dụ chạy từ thư mục này; thay đường dẫn `DATA4`, `COCO`, `OUT4` bằng đường dẫn thực:

```text
python finetune_that_ai.py --pretrained COCO/yolo26m.pt --data-root DATA4 --output OUT4 --check-only
python finetune_that_ai.py --pretrained COCO/yolo26m.pt --data-root DATA4 --output OUT4
```

Code tự nhận bộ đầy đủ để chọn đúng 700+700 hoặc bản đã chọn có sẵn 700+700 trong manifest, giữ thứ tự của bản đã chuẩn bị. Checkpoint nghiên cứu ở giai đoạn này cần `finetune_that_ai.py` trên Python import path khi nạp.

## 2. Học tiếp thêm tàu

`SOURCE4` là checkpoint 4 lớp gốc từ bước 1 hoặc bản nguồn đã chọn trước đó:

```text
python finetune_5_lop.py --source SOURCE4/best.pt --pretrained COCO/yolo26m.pt --data-root DATA5 --output OUT5 --check-only
python finetune_5_lop.py --source SOURCE4/best.pt --pretrained COCO/yolo26m.pt --data-root DATA5 --output OUT5 --train
```

Preflight kiểm tra dữ liệu, mask, gradient, checkpoint khởi tạo và loader trước khi train. Bộ ảnh thiếu nhãn một phần/vùng bỏ qua bắt buộc dùng adapter trong `railway_five_class.py`; YOLO loader mặc định không tự xử lý các quy ước đó. Các phép kiểm tra của preflight gắn với bộ dữ liệu đã rà, không phải bộ kiểm tra tổng quát cho dataset bất kỳ.

Mỗi lần train tạo thư mục mới; không chép checkpoint vào model mặc định của web. Khi resume, dùng checkpoint nghiên cứu còn optimizer, cùng môi trường và bộ dữ liệu; không dùng bản `.pt` suy luận đã strip optimizer.

## 3. Gộp car và xuất checkpoint độc lập

```text
python export_portable_car.py --source SOURCE5/best.pt --output EXPORT4
```

Có thể thêm `--data-root DATA5` để kiểm tra thêm đầu ra trên ảnh validation và `--expected-sha256 HASH` để xác nhận đầu vào. Đầu ra:

- `EXPORT4/models/yolo26m-5class-portable.pt`: đối chứng giữ nguyên 5 lớp.
- `EXPORT4/models/yolo26m-4class-car.pt`: 0 `person`, 1 `two_wheeler`, 2 `car` (gồm xe lớn), 3 `train`.
- `EXPORT4/export-verification.json`: kiểm tra tensor, phép tính và nạp trong Python cách ly.

Phép gộp là maximum logit trước top-k, không cộng confidence hoặc thêm NMS. File cuối không cần các module nghiên cứu khi suy luận. Định nghĩa lớp tùy chỉnh được thay bằng các lớp PyTorch có sẵn; đây không phải lượt train đầu phân loại chuẩn mới.

## 4. Chuyển dữ liệu sang 4 nhãn mới

```text
python prepare_car_four_class.py --data-root DATA5 --output DATA4_CAR
```

Có thể thêm `--road-test ROAD405` nếu cần chuyển bộ test người/xe. Ảnh được sao chép, mọi tọa độ và số box giữ nguyên; mask của hai nhóm xe được gộp bằng phép hợp. Thư mục nguồn và đầu ra phải tách biệt. Dữ liệu sau bước này dành cho nghiên cứu huấn luyện 4 nhãn mới; không đưa thẳng vào trainer 5 lớp ở bước 2.

## Phạm vi bản bàn giao

Thư mục này là code fine-tune/gộp đã dùng cho dòng model được công bố, không phải lượt YOLO chuẩn 20 epoch thử nghiệm hoặc cấu hình học toàn mạng theo Lâm. Việc bổ sung code không bắt đầu một lượt train mới và không đổi checkpoint đang bàn giao.

Kiểm tra trên bản code này: lệnh `--help` và import; xác nhận 1.400 ảnh train/125 ảnh validation cùng thứ tự chọn dữ liệu; preflight 5 lớp với loss/gradient/coverage/mask; chuyển mẫu nhãn và mask khớp bản đã chuẩn bị; xuất lại model có toàn bộ tensor và đầu ra khớp checkpoint 4 lớp đã công bố. Các kiểm tra không chạy optimizer hoặc epoch huấn luyện mới.

Để tích hợp vào web, chỉ cần checkpoint cuối và thư viện phù hợp; xem [hướng dẫn model](../../../models/minh_yolo26m_4class_car/README.md). Các file code trong `research/` không phải phụ thuộc suy luận của model cuối.
