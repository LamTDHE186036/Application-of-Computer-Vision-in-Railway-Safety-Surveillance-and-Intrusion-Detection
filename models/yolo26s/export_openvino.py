"""
Công cụ Chuyển đổi và Tối ưu hóa Mô hình YOLO26s sang định dạng Intel OpenVINO
Chuyển đổi trọng số PyTorch (.pt) -> OpenVINO Intermediate Representation (.xml / .bin)
Hỗ trợ tối ưu hóa FP16/FP32, gộp toán tử (Layer Fusion), tận dụng AVX-512 & Intel Iris Xe iGPU.
"""

import os
import argparse
import shutil
from ultralytics import YOLO

def export_model_to_openvino(source_pt, output_dir=None, half=True, imgsz=640):
    """
    Xuất mô hình PyTorch YOLO sang định dạng OpenVINO tối ưu hóa
    :param source_pt: Đường dẫn file trọng số .pt
    :param output_dir: Thư mục lưu kết quả xuất
    :param half: Sử dụng chuẩn số thực nửa chính xác FP16 để tăng tốc độ và giảm 50% dung lượng
    :param imgsz: Kích thước ảnh đầu vào (mặc định 640)
    """
    if not source_pt or not os.path.exists(source_pt):
        print(f"[!] Lỗi: Không tìm thấy file mô hình PyTorch tại: {source_pt}")
        return None

    print("=" * 70)
    print(" CÔNG CỤ TỐI ƯU HÓA & XUẤT MÔ HÌNH YOLO26s SANG INTEL OPENVINO")
    print("=" * 70)
    print(f"[*] Trọng số PyTorch gốc: {source_pt}")
    print(f"[*] Kích thước đầu vào : {imgsz}x{imgsz}")
    print(f"[*] Chuẩn tối ưu hóa   : {'FP16 (Half Precision)' if half else 'FP32 (Single Precision)'}")

    # Nạp mô hình PyTorch
    model = YOLO(source_pt)

    # Thực hiện export sang OpenVINO
    print("[*] Đang biên dịch đồ thị tính toán và tối ưu hóa toán tử (Graph Optimization)...")
    exported_dir = model.export(format="openvino", half=half, imgsz=imgsz)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        final_dir = os.path.abspath(output_dir)
        if os.path.abspath(exported_dir) != final_dir:
            print(f"[*] Đang đồng bộ mô hình đã xuất vào: {final_dir}")
            # Dọn dẹp các file mô hình cũ trong thư mục đích trước khi sao chép
            for item in os.listdir(final_dir):
                target_path = os.path.join(final_dir, item)
                if item.endswith((".xml", ".bin", ".yaml")) or os.path.isdir(target_path):
                    try:
                        if os.path.isdir(target_path):
                            shutil.rmtree(target_path)
                        else:
                            os.remove(target_path)
                    except Exception as e:
                        print(f"[!] Cảnh báo không thể xóa file cũ {item}: {e}")
            for item in os.listdir(exported_dir):
                s = os.path.join(exported_dir, item)
                d = os.path.join(final_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            # Dọn dẹp thư mục tạm do Ultralytics sinh ra
            try:
                shutil.rmtree(exported_dir)
            except Exception:
                pass
            exported_dir = final_dir

    print("\n" + "=" * 70)
    print(" XUẤT MÔ HÌNH OPENVINO HOÀN TẤT THÀNH CÔNG!")
    print(f"-> Thư mục chứa OpenVINO IR: {exported_dir}")
    print("-> Bao gồm file cấu trúc mạng (.xml) và file trọng số nhị phân (.bin)")
    print("=" * 70)
    return exported_dir

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    # Đường dẫn mô hình mặc định và danh sách tìm kiếm
    default_pt = os.path.join(current_dir, "model_pytorch", "train-railway-yolo26s-all.pt")
    if not os.path.exists(default_pt):
        candidates = [
            os.path.join(project_root, "research", "finetune", "train_yolo26s_all", "train-railway-yolo26s-all.pt"),
            os.path.join(project_root, "research", "finetune", "train_yolo26s_all", "weights", "best.pt"),
            os.path.join(current_dir, "model_pytorch", "yolo26s.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                default_pt = c
                break
        else:
            pt_dir = os.path.join(current_dir, "model_pytorch")
            if os.path.exists(pt_dir):
                pts = [os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith(".pt")]
                if pts:
                    default_pt = pts[0]

    default_out = os.path.join(current_dir, "model_openvino")

    parser = argparse.ArgumentParser(description="Tối ưu hóa mô hình YOLO26s sang Intel OpenVINO")
    parser.add_argument("--weights", type=str, default="",
                        help=f"Đường dẫn file .pt nguồn (mặc định: {default_pt})")
    parser.add_argument("--output", type=str, default="", 
                        help=f"Thư mục xuất kết quả (mặc định: {default_out})")
    parser.add_argument("--fp16", dest="half", action="store_true", default=True, help="Tối ưu hóa chuẩn FP16 (mặc định)")
    parser.add_argument("--fp32", dest="half", action="store_false", help="Sử dụng chuẩn FP32 thay vì FP16")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước khung hình")
    args = parser.parse_args()

    # Mặc định lấy trọng số YOLO26s hiện tại
    weights = args.weights
    if not weights:
        candidates = [
            default_pt,
            os.path.join(current_dir, "model_pytorch", "train-railway-yolo26s-all.pt"),
            os.path.join(project_root, "research", "finetune", "train_yolo26s_all", "train-railway-yolo26s-all.pt"),
            os.path.join(project_root, "research", "finetune", "train_yolo26s_all", "weights", "best.pt"),
            os.path.join(current_dir, "model_pytorch", "yolo26s.pt"),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                weights = os.path.abspath(c)
                break

    output_dir = args.output if args.output else default_out

    export_model_to_openvino(source_pt=weights, output_dir=output_dir, half=args.half, imgsz=args.imgsz)
