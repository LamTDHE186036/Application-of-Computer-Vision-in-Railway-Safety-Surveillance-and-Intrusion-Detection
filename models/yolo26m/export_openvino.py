"""
Công cụ Chuyển đổi và Tối ưu hóa Mô hình YOLO26m sang định dạng Intel OpenVINO
Chuyển đổi trọng số PyTorch (.pt) -> OpenVINO Intermediate Representation (.xml / .bin)
Hỗ trợ tối ưu hóa FP16/FP32, gộp toán tử (Layer Fusion), tận dụng AVX-512 & Intel Iris Xe iGPU.
"""

import os
import sys
import argparse
import shutil
import tempfile

# Đảm bảo xuất tiếng Việt ổn định trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đảm bảo Ultralytics luôn có thư mục cấu hình ghi được
if not os.environ.get("YOLO_CONFIG_DIR"):
    _tmp_cfg = os.path.join(tempfile.gettempdir(), "railsafe_ultralytics_cfg")
    try:
        os.makedirs(_tmp_cfg, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = _tmp_cfg
    except Exception:
        _local_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ultralytics_cfg")
        os.makedirs(_local_cfg, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = _local_cfg

from ultralytics import YOLO


def export_model_to_openvino(source_pt, output_dir=None, half=True, imgsz=640):
    """
    Xuất mô hình PyTorch YOLO26m sang định dạng OpenVINO tối ưu hóa.
    :param source_pt: Đường dẫn file trọng số .pt
    :param output_dir: Thư mục lưu kết quả xuất
    :param half: Sử dụng chuẩn số thực nửa chính xác FP16 để tăng tốc độ và giảm 50% dung lượng
    :param imgsz: Kích thước ảnh đầu vào (mặc định 640)
    """
    if not source_pt or not os.path.exists(source_pt):
        print(f"[!] Lỗi: Không tìm thấy file mô hình PyTorch tại: {source_pt}")
        return None

    print("=" * 70)
    print(" CÔNG CỤ TỐI ƯU HÓA & XUẤT MÔ HÌNH YOLO26m SANG INTEL OPENVINO")
    print("=" * 70)
    print(f"[*] Trọng số PyTorch gốc: {source_pt}")
    print(f"[*] Kích thước đầu vào : {imgsz}x{imgsz}")
    print(f"[*] Chuẩn tối ưu hóa   : {'FP16 (Half Precision)' if half else 'FP32 (Single Precision)'}")

    model = YOLO(source_pt)

    print("[*] Đang biên dịch đồ thị tính toán và tối ưu hóa toán tử (Graph Optimization)...")
    exported_dir = model.export(format="openvino", half=half, imgsz=imgsz)

    if output_dir:
        final_dir = os.path.abspath(output_dir)
        # Nếu trước đó có tệp rỗng trùng tên thư mục đích thì xóa đi để tạo thư mục
        if os.path.isfile(final_dir):
            os.remove(final_dir)
        os.makedirs(final_dir, exist_ok=True)

        if os.path.abspath(exported_dir) != final_dir:
            print(f"[*] Đang đồng bộ mô hình đã xuất vào: {final_dir}")
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
            try:
                shutil.rmtree(exported_dir)
            except Exception:
                pass
            exported_dir = final_dir

    print("\n" + "=" * 70)
    print(" XUẤT MÔ HÌNH OPENVINO HOÀN TẤT THÀNH CÔNG!")
    print(f"-> Thư mục chứa OpenVINO IR: {exported_dir}")
    print("-> Bao gồm file cấu trúc mạng (.xml), trọng số nhị phân (.bin) và metadata.yaml")
    print("=" * 70)
    return exported_dir


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    default_pt = os.path.join(current_dir, "model_pytorch", "yolo26m.pt")
    if not os.path.exists(default_pt):
        pt_dir = os.path.join(current_dir, "model_pytorch")
        if os.path.exists(pt_dir):
            pts = [os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith(".pt")]
            if pts:
                default_pt = pts[0]

    default_out = os.path.join(current_dir, "model_openvino")

    parser = argparse.ArgumentParser(description="Tối ưu hóa mô hình YOLO26m sang Intel OpenVINO")
    parser.add_argument("--weights", type=str, default="", help=f"Đường dẫn file .pt nguồn (mặc định: {default_pt})")
    parser.add_argument("--output", type=str, default="", help=f"Thư mục xuất kết quả (mặc định: {default_out})")
    parser.add_argument("--fp16", dest="half", action="store_true", default=True, help="Tối ưu hóa chuẩn FP16 (mặc định)")
    parser.add_argument("--fp32", dest="half", action="store_false", help="Sử dụng chuẩn FP32 thay vì FP16")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước khung hình")
    args = parser.parse_args()

    weights = args.weights if args.weights else (os.path.abspath(default_pt) if os.path.exists(default_pt) else "")
    output_dir = args.output if args.output else default_out

    export_model_to_openvino(source_pt=weights, output_dir=output_dir, half=args.half, imgsz=args.imgsz)
