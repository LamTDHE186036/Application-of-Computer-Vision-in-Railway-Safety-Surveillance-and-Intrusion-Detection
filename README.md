# 🚀 YOLO Vision Studio: Real-time WebRTC, Video & Image AI

Ứng dụng web trực quan, hiện đại kết hợp giữa **Gradio 5**, **Ultralytics YOLO (YOLO11 & YOLOv8)**, **WebRTC (FastRTC)** và **AI Text-to-Speech (gTTS)** với 3 chế độ chuyên biệt:

---

## 🎯 3 Chế Độ Hoạt Động

### 1. 📹 Chế độ 1: Webcam Realtime (WebRTC - Không Giật Lag)
- **Truyền phát thời gian thực siêu tốc:** Sử dụng giao thức WebRTC (UDP ngang hàng), độ trễ cực thấp **30ms – 50ms**, không bị tích tụ lag hay đơ hình như HTTP thông thường.
- **Khung hình đôi theo yêu cầu:** 
  - **Bên Trái:** Đánh Bounding Box, hiển thị nhãn và tên các vật thể đang có trong tầm nhìn.
  - **Bên Phải:** Luồng Webcam gốc thời gian thực.
- **HUD thông minh:** Hiển thị trực tiếp FPS thực tế và danh sách vật thể đang nhận diện.

### 2. 🎬 Chế độ 2: Phân Tích Video Tải Lên (Upload Video)
- **Tải lên video từ laptop/máy tính:** Hỗ trợ mọi định dạng MP4, AVI, MOV, MKV...
- **Bên Trái:** Bảng thống kê chi tiết các vật thể xuất hiện trong video (Tên tiếng Việt, số lần xuất hiện theo frames, số lượng tối đa cùng lúc, độ tin cậy trung bình).
- **Bên Phải:** Video kết quả đã vẽ Bounding Boxes sắc nét.
- **Chuẩn H.264 qua PyAV:** Video xuất ra tương thích 100% với mọi trình duyệt web (Chrome, Edge, Firefox, Safari) mà không bị lỗi màn hình đen.
- **Tùy chọn Skip Frame:** Tăng tốc xử lý video dài gấp 2 đến 4 lần.

### 3. 🖼️ Chế độ 3: Nhận Diện Ảnh & Thuyết Minh Giọng Nói AI (Image & Voice)
- **Tải lên ảnh bất kỳ** từ máy tính hoặc clipboard.
- **Đánh Bounding Boxes** chính xác từng đối tượng.
- **Thuyết minh giọng nói AI (gTTS Tiếng Việt):** Tự động phát âm thanh đọc to rõ danh sách đối tượng có trong bức ảnh (ví dụ: *"Trong bức ảnh này phát hiện thấy 2 người và 1 chiếc điện thoại"*).
- **Bảng chi tiết:** Tọa độ [x1, y1, x2, y2], độ tin cậy và class tương ứng.

---

## 💡 Điểm nổi bật: Vì sao WebRTC không bị giật lag?

| Đặc điểm | Webcam thông thường (`gr.Image(streaming=True)`) | Webcam WebRTC (`fastrtc.WebRTC`) |
| :--- | :--- | :--- |
| **Giao thức** | HTTP POST qua TCP | UDP / RTP ngang hàng (Peer-to-peer) |
| **Độ trễ (Latency)** | **1.000ms – 3.000ms** (Bị delay tích tụ dần) | **30ms – 50ms** (Gần như tức thì) |
| **Hiện tượng nghẽn** | Frame cũ dồn ứ hàng đợi TCP -> đơ hình | Bỏ qua gói tin quá hạn, luôn nhận frame mới nhất |
| **Băng thông** | Truyền chuỗi Base64 nặng nề | Nén video streaming thời gian thực bằng phần cứng |

---

## 🏃 Hướng Dẫn Khởi Chạy

### Cách 1: Chạy bằng dòng lệnh
```bash
python app.py
```

### Cách 2: Chạy trên Windows
Click đúp chuột vào file `run.bat`.

Sau khi chạy, mở trình duyệt truy cập:
👉 **`http://127.0.0.1:7860`**
