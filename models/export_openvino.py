import os
import shutil
from ultralytics import YOLO

def export_pt_to_openvino(pt_path=None, imgsz=640):
    """
    Chuyển đổi mô hình YOLO định dạng PyTorch (.pt) sang OpenVINO tối ưu hóa phần cứng
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    # 1. Tự động xác định file .pt nguồn
    if not pt_path:
        candidates = [
            os.path.join(current_dir, "yolo26s", "model.pt"),
            os.path.join(project_root, "research", "finetune", "train_yolo26s", "train-railway-yolo26s.pt"),
            os.path.join(current_dir, "train-railway-yolo26s.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                pt_path = c
                break

    if not pt_path or not os.path.exists(pt_path):
        print(f"[!] Lỗi: Không tìm thấy file trọng số .pt nào để chuyển đổi.")
        return False

    target_dir = os.path.join(current_dir, "yolo26s")
    os.makedirs(target_dir, exist_ok=True)
    target_pt = os.path.join(target_dir, "model.pt")

    # Nếu file nguồn chưa nằm trong models/yolo26s/model.pt, sao chép vào đó
    if os.path.abspath(pt_path) != os.path.abspath(target_pt):
        print(f"[*] Sao chép {pt_path} -> {target_pt}")
        shutil.copy2(pt_path, target_pt)

    print("=" * 65)
    print(f"[*] TIẾN HÀNH XUẤT MÔ HÌNH SANG OPENVINO")
    print(f"    - File PyTorch nguồn: {target_pt}")
    print(f"    - Kích thước ảnh: {imgsz}x{imgsz}")
    print("=" * 65)

    # 2. Nạp mô hình YOLO
    model = YOLO(target_pt)

    # 3. Xuất sang OpenVINO
    # Ultralytics tự động tạo thư mục chứa .xml và .bin
    exported_path = model.export(format="openvino", imgsz=imgsz, half=False)

    # 4. Chuẩn hóa tên thư mục thành 'model_openvino' để hệ thống web nhận diện
    final_openvino_dir = os.path.join(target_dir, "model_openvino")
    if os.path.exists(exported_path) and os.path.abspath(exported_path) != os.path.abspath(final_openvino_dir):
        if os.path.exists(final_openvino_dir):
            shutil.rmtree(final_openvino_dir)
        shutil.move(exported_path, final_openvino_dir)

    print("\n" + "=" * 65)
    print(" CHUYỂN ĐỔI OPENVINO THÀNH CÔNG!")
    print(f"-> Thư mục OpenVINO hoàn thiện: {final_openvino_dir}")
    print(f"-> File trọng số PyTorch gốc:   {target_pt}")
    print("=" * 65)
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chuyển đổi YOLO .pt sang OpenVINO")
    parser.add_argument("--pt", type=str, default=None, help="Đường dẫn tới file .pt (mặc định tự tìm yolo26s)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (mặc định: 640)")
    args = parser.parse_args()

    export_pt_to_openvino(pt_path=args.pt, imgsz=args.imgsz)
