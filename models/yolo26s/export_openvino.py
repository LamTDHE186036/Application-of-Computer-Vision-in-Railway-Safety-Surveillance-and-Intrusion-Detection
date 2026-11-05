"""
Thực nghiệm 3: Công cụ Chuyển đổi và Tối ưu hóa Mô hình sang định dạng Intel OpenVINO
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
    if not os.path.exists(source_pt):
        print(f"[!] Lỗi: Không tìm thấy file mô hình PyTorch tại: {source_pt}")
        return None

    print("=" * 70)
    print(" CÔNG CỤ TỐI ƯU HÓA & XUẤT MÔ HÌNH SANG INTEL OPENVINO")
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
        print(f"[*] Đang đồng bộ mô hình đã xuất vào: {final_dir}")
        for item in os.listdir(exported_dir):
            s = os.path.join(exported_dir, item)
            d = os.path.join(final_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        exported_dir = final_dir

    print("\n" + "=" * 70)
    print(" XUẤT MÔ HÌNH OPENVINO HOÀN TẤT THÀNH CÔNG!")
    print(f"-> Thư mục chứa OpenVINO IR: {exported_dir}")
    print("-> Bao gồm file cấu trúc mạng (.xml) và file trọng số nhị phân (.bin)")
    print("=" * 70)
    return exported_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tối ưu hóa mô hình YOLO sang Intel OpenVINO")
    parser.add_argument("--weights", type=str, default="",
                        help="Đường dẫn file .pt nguồn (mặc định lấy research/finetune/train_yolo26s_all)")
    parser.add_argument("--output", type=str, default="", 
                        help="Thư mục xuất kết quả (mặc định lưu vào models/yolo26s/model_openvino)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Tối ưu hóa chuẩn FP16")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước khung hình")
    args = parser.parse_args()

    # Mặc định lấy từ train_yolo26s_all
    weights = args.weights
    if not weights:
        candidates = [
            "research/finetune/train_yolo26s_all/train-railway-yolo26s-all.pt",
            "models/yolo26s/model_pytorch/train-railway-yolo26s-all.pt",
            "research/finetune/train_yolo26s/train-railway-yolo26s.pt"
        ]
        for c in candidates:
            if os.path.exists(c):
                weights = c
                break

    output_dir = args.output
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(__file__), "model_openvino")

    export_model_to_openvino(source_pt=weights, output_dir=output_dir, half=args.fp16, imgsz=args.imgsz)

