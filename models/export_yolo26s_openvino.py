import os
import shutil
from ultralytics import YOLO

def convert_and_save_model():
    # Xác định thư mục gốc dự án (Web_flask) để script chạy được từ bất kỳ đâu
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 1. Định nghĩa đường dẫn file nguồn và thư mục đích
    source_pt = os.path.join(base_dir, "finetune", "train_yolo26s", "train-railway-yolo26s.pt")
    target_models_dir = os.path.join(base_dir, "models")
    target_pt = os.path.join(target_models_dir, "train-railway-yolo26s.pt")

    # Kiểm tra file nguồn có tồn tại không
    if not os.path.exists(source_pt):
        print(f"[!] Lỗi: Không tìm thấy file mô hình tại {source_pt}")
        return

    # Đảm bảo thư mục models đã được tạo
    os.makedirs(target_models_dir, exist_ok=True)

    # 2. Sao chép file .pt vào thư mục models làm bản dự phòng
    print(f"[*] Đang sao chép file .pt vào thư mục models...")
    shutil.copy2(source_pt, target_pt)
    print(f"[+] Đã lưu bản gốc PyTorch: {target_pt}")

    # 3. Nạp mô hình YOLO từ file trong thư mục models
    print("[*] Đang nạp mô hình YOLO26s để chuẩn bị chuyển đổi...")
    model = YOLO(target_pt)

    # 4. Xuất mô hình sang định dạng OpenVINO (độ phân giải 640x640 chuẩn YOLO)
    print("[*] Đang tối ưu hóa và xuất sang định dạng OpenVINO (vui lòng chờ)...")
    export_path = model.export(format="openvino", imgsz=640)

    print("\n" + "="*60)
    print(" QUÁ TRÌNH CHUYỂN ĐỔI YOLO26s THÀNH CÔNG!")
    print(f"-> Thư mục OpenVINO hoàn thiện: {export_path}")
    print(f"-> File trọng số PyTorch dự phòng: {target_pt}")
    print("="*60)

if __name__ == "__main__":
    convert_and_save_model()
