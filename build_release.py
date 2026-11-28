import os
import sys
import shutil
import subprocess

# Đảm bảo xuất tiếng Việt có dấu ổn định trên cửa sổ dòng lệnh Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

APP_VERSION = "1.3.0"

def build():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(base_dir, "dist")
    target_dir = os.path.join(dist_dir, "RailwaySafety")

    print("=" * 70)
    print(f"   TIẾN HÀNH ĐÓNG GÓI SẢN PHẨM V{APP_VERSION} BẰNG PYINSTALLER")
    print("=" * 70)

    # 0. Kiểm tra cấu hình AI Runtime (ONNX DirectML + OpenVINO + PyTorch) trước khi đóng gói
    try:
        import torch
        torch_ver = getattr(torch, "__version__", "unknown")
        torch_cuda = getattr(torch.version, "cuda", None)
        if torch_cuda:
            print(f"[!] LƯU Ý: PyTorch đang là bản CUDA ({torch_ver}) -> gói build sẽ nặng thêm ~2.5 GB.")
            print("    Nếu muốn bộ cài gọn ~400MB (1 file .exe, vẫn chạy GPU rời qua ONNX DirectML), hãy cài PyTorch CPU:")
            print("    pip install --force-reinstall --no-deps torch==2.4.1+cpu torchvision==0.19.1+cpu --index-url https://download.pytorch.org/whl/cpu")
        else:
            print(f"[*] PyTorch đóng gói: {torch_ver} (gọn nhẹ, không kèm 3.4GB CUDA DLL)")
    except Exception as e:
        print(f"[!] Không thể kiểm tra phiên bản PyTorch: {e}")

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_dml = "DmlExecutionProvider" in providers
        print(f"[*] ONNX Runtime    : {ort.__version__} | DirectML GPU: {'CÓ (NVIDIA RTX / AMD / Intel DX12)' if has_dml else 'KHÔNG'}")
        if not has_dml and "--require-directml" in sys.argv:
            print("[!] Hủy đóng gói: thiếu onnxruntime-directml (DmlExecutionProvider).")
            sys.exit(1)
    except Exception as e:
        print(f"[!] Cảnh báo ONNX Runtime DirectML chưa sẵn sàng: {e}")
        if "--require-directml" in sys.argv:
            sys.exit(1)

    # 1. Chạy PyInstaller
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "RailwaySafety.spec"]
    print(f"[*] Đang chạy lệnh: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=base_dir)
    if res.returncode != 0:
        print("[!] Lỗi xảy ra trong quá trình đóng gói bằng PyInstaller!")
        sys.exit(res.returncode)

    print("\n[*] PyInstaller hoàn tất! Tiến hành chuẩn bị thư mục phân phối...")

    # 2. Sao chép đồng bộ cả 2 mô hình AI (yolo26s cho Intel iGPU/CPU và yolo26m cho NVIDIA CUDA) sang dist/RailwaySafety/models/
    src_models = os.path.join(base_dir, "models")
    dst_models = os.path.join(target_dir, "models")
    os.makedirs(dst_models, exist_ok=True)

    for model_folder in ("yolo26s", "yolo26m"):
        src_m = os.path.join(src_models, model_folder)
        dst_m = os.path.join(dst_models, model_folder)
        if os.path.exists(src_m):
            print(f"[*] Đang sao chép mô hình AI {model_folder} sang: {dst_m}")
            if os.path.exists(dst_m):
                shutil.rmtree(dst_m)
            shutil.copytree(
                src_m,
                dst_m,
                ignore=shutil.ignore_patterns("*.py", "*.pyc", "__pycache__", ".git*", ".ultralytics_cfg")
            )

    # 3. Sao chép configs/ sang dist/RailwaySafety/configs/ (luôn ghi đè để cập nhật cấu hình mới nhất)
    src_configs = os.path.join(base_dir, "configs")
    dst_configs = os.path.join(target_dir, "configs")
    if os.path.exists(src_configs):
        print(f"[*] Đang sao chép cấu hình configs sang: {dst_configs}")
        os.makedirs(dst_configs, exist_ok=True)
        for item in os.listdir(src_configs):
            if item.endswith(('.json', '.yaml', '.yml')):
                s = os.path.join(src_configs, item)
                d = os.path.join(dst_configs, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)

    # 4. Sao chép audio_alerts/ sang dist/RailwaySafety/audio_alerts/
    src_audio = os.path.join(base_dir, "audio_alerts")
    dst_audio = os.path.join(target_dir, "audio_alerts")
    if os.path.exists(src_audio):
        print(f"[*] Đang sao chép audio_alerts sang: {dst_audio}")
        if os.path.exists(dst_audio):
            shutil.rmtree(dst_audio)
        shutil.copytree(
            src_audio,
            dst_audio,
            ignore=shutil.ignore_patterns("*.py", "*.pyc", "__pycache__", ".git*")
        )

    # 5. Sao chép ffmpeg.exe sang dist/RailwaySafety/ để hỗ trợ chuyển đổi video chuẩn web H.264
    src_ffmpeg = os.path.join(base_dir, "ffmpeg.exe")
    dst_ffmpeg = os.path.join(target_dir, "ffmpeg.exe")
    if not os.path.exists(src_ffmpeg):
        try:
            import imageio_ffmpeg
            cand = imageio_ffmpeg.get_ffmpeg_exe()
            if cand and os.path.exists(cand):
                src_ffmpeg = cand
        except Exception:
            pass
    if not os.path.exists(src_ffmpeg):
        cand = shutil.which("ffmpeg")
        if cand and os.path.exists(cand):
            src_ffmpeg = cand

    if src_ffmpeg and os.path.exists(src_ffmpeg):
        print(f"[*] Đang sao chép ffmpeg.exe sang: {dst_ffmpeg}")
        shutil.copy2(src_ffmpeg, dst_ffmpeg)
    else:
        print("[!] CẢNH BÁO: Không tìm thấy ffmpeg.exe để sao chép vào gói phân phối!")

    # 5.5 Kiểm tra và tự động tải vc_redist.x64.exe nếu thiếu (phục vụ tạo bộ cài Inno Setup)
    vc_path = os.path.join(base_dir, "vc_redist.x64.exe")
    if not os.path.exists(vc_path):
        print("[*] Đang tải vc_redist.x64.exe từ Microsoft cho Inno Setup...")
        try:
            import urllib.request
            url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
            urllib.request.urlretrieve(url, vc_path)
            print("[*] Đã tải xong vc_redist.x64.exe!")
        except Exception as e:
            print(f"[!] Cảnh báo: Không thể tự động tải vc_redist.x64.exe ({e}). Vui lòng tải thủ công vào thư mục gốc nếu dùng Inno Setup.")
    else:
        print("[*] vc_redist.x64.exe đã sẵn sàng cho Inno Setup.")

    # 6. Tạo cấu trúc thư mục data/ ngang hàng với RailwaySafety.exe (không sao chép video cũ để tiết kiệm dung lượng)
    dst_data = os.path.join(target_dir, "data")
    dst_recordings = os.path.join(dst_data, "recordings")
    dst_logs = os.path.join(dst_data, "intrusion_logs")
    os.makedirs(dst_recordings, exist_ok=True)
    os.makedirs(dst_logs, exist_ok=True)
    print(f"[*] Đã tạo thư mục data ngang hàng tại: {dst_data}")

    # 7. Tạo tệp Khoi_Dong.bat giữ lại cửa sổ console để hiển thị thông tin và tránh tự động thoát
    bat_content = f"""@echo off
chcp 65001 >nul
title Hệ Thống Giám Sát An Toàn Đường Sắt v{APP_VERSION}
cd /d "%~dp0"
echo ========================================================
echo    HỆ THỐNG GIÁM SÁT AN TOÀN ĐƯỜNG SẮT (v{APP_VERSION})
echo ========================================================
echo Đang khởi chạy hệ thống...
RailwaySafety.exe
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================================
    echo [THÔNG BÁO] Chương trình đã dừng lại với mã lỗi: %ERRORLEVEL%
    echo ========================================================
    pause
)
"""
    bat_path = os.path.join(target_dir, "Khoi_Dong.bat")
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)

    # 8. Tạo tệp README_HUONG_DAN.txt cho người dùng cuối
    readme_content = f"""========================================================================
   HỆ THỐNG GIÁM SÁT AN TOÀN ĐƯỜNG SẮT VÀ CẢNH BÁO VI PHẠM (V{APP_VERSION})
========================================================================

1. CÁCH KHỞI ĐỘNG:
   - Nhấp đúp vào tệp "Khoi_Dong.bat" hoặc "RailwaySafety.exe".
   - Máy chủ sẽ tự động khởi chạy và mở trình duyệt tại: http://127.0.0.1:5000/
   - Cửa sổ Console được giữ lại để theo dõi trạng thái và nhật ký hệ thống.

2. XEM VIDEO VÀ LỊCH SỬ VI PHẠM TRỰC TIẾP TRÊN MÁY:
   - Thư mục "data" nằm ngay cạnh tệp "RailwaySafety.exe".
   - Video vi phạm: nằm tại "data/recordings/{{ngày-tháng-năm}}/"
     -> Bạn có thể mở trực tiếp bằng Windows Media Player / VLC / Movies & TV.
   - Nhật ký sự kiện vi phạm: nằm tại "data/intrusion_logs/Lich_su_{{ngày-tháng-năm}}.txt".

3. XEM TRÊN GIAO DIỆN WEB:
   - Truy cập mục "Lịch sử" trên thanh menu trái để xem video và danh sách trực tiếp trên web.

4. CẤU HÌNH:
   - Các thiết lập vùng ROI, ngưỡng tin cậy và camera được lưu tại thư mục "configs/".
========================================================================
"""
    readme_path = os.path.join(target_dir, "README_HUONG_DAN.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("\n" + "=" * 70)
    print("   ĐÓNG GÓI THÀNH CÔNG! SẢN PHẨM PHÂN PHỐI SẴN SÀNG TẠI:")
    print(f"   {target_dir}")
    print("=" * 70)

if __name__ == '__main__':
    build()
