import os
import argparse
import shutil
from ultralytics import YOLO

def export_model_to_openvino(model_name="yolo26s", imgsz=640):
    """
    Tự động nạp trọng số YOLO .pt và xuất sang định dạng OpenVINO tối ưu hóa phần cứng
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Bản đồ mô hình nguồn và thư mục đích
    model_mapping = {
        "yolo11s": {
            "source_pt": os.path.join(base_dir, "research", "finetune", "train_yolo11s", "train-railway-yolo11s.pt"),
            "target_dir": os.path.join(base_dir, "models", "yolo11s")
        },
        "yolo26s": {
            "source_pt": os.path.join(base_dir, "research", "finetune", "train_yolo26s", "train-railway-yolo26s.pt"),
            "target_dir": os.path.join(base_dir, "models", "yolo26s")
        },
        "yolo26s-all": {
            "source_pt": os.path.join(base_dir, "research", "finetune", "train_yolo26s_all", "train-railway-yolo26s-all.pt"),
            "target_dir": os.path.join(base_dir, "models", "yolo26s-all")
        }
    }
    
    config = model_mapping.get(model_name)
    if not config:
        print(f"[!] Lỗi: Tên mô hình '{model_name}' không hợp lệ. Các lựa chọn: {list(model_mapping.keys())}")
        return

    source_pt = config["source_pt"]
    target_dir = config["target_dir"]
    target_pt = os.path.join(target_dir, "model.pt")

    # Kiểm tra file nguồn hoặc file pt sẵn có trong thư mục target
    if not os.path.exists(source_pt):
        if os.path.exists(target_pt):
            print(f"[*] Sử dụng file model.pt sẵn có tại: {target_pt}")
            source_pt = target_pt
        else:
            print(f"[!] Lỗi: Không tìm thấy file mô hình tại {source_pt} và {target_pt}")
            return

    os.makedirs(target_dir, exist_ok=True)
    if source_pt != target_pt:
        print(f"[*] Đang sao chép file .pt vào {target_pt}...")
        shutil.copy2(source_pt, target_pt)
        print(f"[+] Đã lưu bản gốc PyTorch: {target_pt}")

    print(f"[*] Đang nạp mô hình {model_name}...")
    model = YOLO(target_pt)

    print(f"[*] Đang tối ưu hóa và xuất sang định dạng OpenVINO ({imgsz}x{imgsz})...")
    export_path = model.export(format="openvino", imgsz=imgsz)

    # Đổi tên thư mục xuất thành model_openvino trong thư mục target nếu cần
    target_openvino = os.path.join(target_dir, "model_openvino")
    if export_path and os.path.exists(export_path) and os.path.abspath(export_path) != os.path.abspath(target_openvino):
        if os.path.exists(target_openvino):
            shutil.rmtree(target_openvino)
        shutil.move(export_path, target_openvino)
        export_path = target_openvino

    print("\n" + "=" * 60)
    print(f" QUÁ TRÌNH CHUYỂN ĐỔI {model_name.upper()} THÀNH CÔNG!")
    print(f"-> Thư mục OpenVINO hoàn thiện: {export_path}")
    print(f"-> File trọng số PyTorch: {target_pt}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xuất mô hình YOLO sang OpenVINO")
    parser.add_argument("--model", type=str, default="yolo26s", choices=["yolo11s", "yolo26s", "yolo26s-all"],
                        help="Tên mô hình cần xuất (mặc định: yolo26s)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (mặc định: 640)")
    args = parser.parse_args()
    
    export_model_to_openvino(model_name=args.model, imgsz=args.imgsz)
