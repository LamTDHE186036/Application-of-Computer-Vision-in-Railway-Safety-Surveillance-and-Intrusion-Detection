# 🚆 Railway Safety Surveillance & Intrusion Detection System
Hệ thống giám sát an toàn và phát hiện xâm phạm đường sắt thời gian thực ứng dụng thị giác máy tính (AI Computer Vision), tối ưu phần cứng OpenVINO (Intel Iris Xe GPU / CPU) và theo dõi đối tượng ByteTrack.

---

## 📁 Cấu Trúc Dự Án (Project Architecture)

```
railway-safety-detection/
│
├── main.py                      # Entry point duy nhất khởi chạy hệ thống
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── README.md                    # Tài liệu hướng dẫn sử dụng và kiến trúc
├── .gitignore                   # Cấu hình bỏ qua tệp tạm và dữ liệu runtime
├── run.bat                      # Launcher khởi chạy nhanh cho Windows
│
├── app/                         # Toàn bộ logic ứng dụng (Modular Application)
│   ├── __init__.py              # Flask Application Factory & Blueprints
│   ├── config.py                # Quản lý đọc/ghi cấu hình & Runtime State
│   │
│   ├── routes/                  # Phân chia routes theo từng domain
│   │   ├── __init__.py          # Đăng ký Blueprints vào Flask app
│   │   ├── stream.py            # Routes streaming, video feed & WebRTC
│   │   ├── history.py           # Routes tra cứu lịch sử & phát lại clip
│   │   └── settings.py          # Routes cài đặt tham số (conf, dwell, ROI)
│   │
│   ├── services/                # Business logic & AI Processing
│   │   ├── __init__.py
│   │   ├── detector.py          # AI detection engine & xử lý frame
│   │   └── tracker.py           # Object tracking & logic vùng ROI (ByteTrack)
│   │
│   └── utils/                   # Tiện ích dùng chung
│       ├── __init__.py
│       └── video_utils.py       # Chuyển đổi H.264, định dạng timeline, vẽ ROI
│
├── models/                      # Chứa trọng số mô hình đã huấn luyện
│   ├── yolo11s/
│   │   ├── model.pt
│   │   └── model_openvino/
│   └── yolo26s/
│       ├── model.pt
│       └── model_openvino/
│
├── configs/                     # Toàn bộ tệp cấu hình tham số YAML/JSON
│   ├── bytetrack.yaml           # Cấu hình thuật toán ByteTrack
│   ├── classes.json             # Danh sách lớp nhận diện được kích hoạt
│   ├── detection.json           # Tham số độ tin cậy (conf), dwell time, mode
│   └── roi.json                 # Tọa độ vùng nguy hiểm đa giác (ROI)
│
├── static/                      # Tài nguyên tĩnh giao diện
│   ├── css/                     # Monitor, history, modal, sidebar styles
│   └── js/                      # App, monitor, history, ROI canvas, settings
│
├── templates/                   # Giao diện web người dùng
│   └── index.html               # Single Page Dashboard điều khiển trung tâm
│
├── data/                        # Dữ liệu runtime ghi hình (gitignored)
│   ├── recordings/              # Video các sự kiện vi phạm (.mp4 H.264)
│   └── intrusion_logs/          # Nhật ký sự kiện dạng văn bản (.txt)
│
├── scripts/                     # Script tiện ích & đo đạc hiệu năng
│   ├── export_openvino.py       # Xuất mô hình PyTorch sang OpenVINO
│   └── benchmark_model.py       # Đo đạc độ trễ (latency) và FPS suy luận
│
└── research/                    # Thí nghiệm & huấn luyện chuyên sâu
    └── finetune/                # Dữ liệu & kết quả finetune các phiên bản YOLO
```

---

## 🎯 Các Tính Năng Nổi Bật

1. **Nhận diện xâm phạm thời gian thực (Real-time Intrusion Detection):**
   - Hỗ trợ cả Webcam trực tiếp và Video tải lên.
   - Bounding Box và nhãn hiển thị trực tiếp, bám sát đối tượng không bị trễ hay lag.
   - Lọc tàu hỏa thông minh: Tàu hỏa lưu thông trên ray được định danh hợp lệ, không gây báo động giả.

2. **Cơ chế vùng nguy hiểm đa giác (Custom Polygon ROI):**
   - Cho phép người dùng vẽ trực quan vùng cảnh báo bất kỳ trên canvas.
   - Lưu trữ tọa độ chuẩn hóa (0.0 -> 1.0) độc lập với độ phân giải video.

3. **2 Chế độ vận hành đường sắt chuyên biệt:**
   - **Chế độ 1 (Tàu sắp tới):** Cảnh báo tức thì ngay khi phát hiện người/phương tiện trong vùng ROI (với bộ lọc chống nhiễu 0.5s).
   - **Chế độ 2 (Không có tàu):** Chỉ phát cảnh báo khi người hoặc phương tiện dừng/đỗ quá thời gian cho phép (Dwell Time).

4. **Hộp đen ghi hình thông minh (Smart Pre-roll & Post-roll):**
   - Bộ đệm xoay vòng (Ring buffer) lưu trước 3 giây trước khi xảy ra vi phạm.
   - Tiếp tục ghi hình 3 giây sau khi đối tượng rời khỏi vùng nguy hiểm.
   - Tự động nén và chuẩn hóa H.264 (faststart) để xem lại mượt mà trên mọi trình duyệt web.

5. **Trung tâm tra cứu lịch sử sự kiện (Audit Log & Video Player):**
   - Quản lý nhật ký theo từng ngày.
   - Xem lại clip vi phạm tương ứng với từng mốc thời gian vi phạm với thanh timeline thông minh.

---

## 🏃 Hướng Dẫn Khởi Chạy

### Cách 1: Khởi chạy bằng script Windows
Nhấp đúp chuột vào tệp `run.bat`.

### Cách 2: Khởi chạy bằng dòng lệnh
```bash
python main.py
```

Sau khi khởi chạy, hệ thống sẽ tự động mở trình duyệt tại:
👉 **`http://127.0.0.1:5000`**
