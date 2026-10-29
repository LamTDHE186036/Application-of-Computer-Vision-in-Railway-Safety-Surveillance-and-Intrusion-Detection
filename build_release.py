import os
import sys
import shutil
import subprocess

def build():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(base_dir, "dist")
    target_dir = os.path.join(dist_dir, "RailwaySafety")

    print("=" * 70)
    print("   TIEN HANH DONG GOI SAN PHAM V1.0.0 BANG PYINSTALLER")
    print("=" * 70)

    # 1. Chạy PyInstaller
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "RailwaySafety.spec"]
    print(f"[*] Dang chay lenh: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=base_dir)
    if res.returncode != 0:
        print("[!] Loi xay ra trong qua trinh dong goi bang PyInstaller!")
        sys.exit(res.returncode)

    print("\n[*] PyInstaller hoan tat! Tien hanh chuan bi thu muc phan phoi...")

    # 2. Sao chep mo hinh AI can thiet (yolo26s gom OpenVINO va PyTorch .pt du phong) sang dist/RailwaySafety/models/
    src_models = os.path.join(base_dir, "models")
    dst_models = os.path.join(target_dir, "models")
    os.makedirs(dst_models, exist_ok=True)

    src_yolo26s = os.path.join(src_models, "yolo26s")
    dst_yolo26s = os.path.join(dst_models, "yolo26s")
    if os.path.exists(src_yolo26s):
        print(f"[*] Dang sao chep mo hinh AI (OpenVINO + PyTorch fallback) sang: {dst_yolo26s}")
        if os.path.exists(dst_yolo26s):
            shutil.rmtree(dst_yolo26s)
        shutil.copytree(
            src_yolo26s,
            dst_yolo26s,
            ignore=shutil.ignore_patterns("*.py", "*.pyc", "__pycache__", ".git*")
        )

    # 3. Sao chep configs/ sang dist/RailwaySafety/configs/ (luon ghi de de cap nhat cau hinh moi nhat)
    src_configs = os.path.join(base_dir, "configs")
    dst_configs = os.path.join(target_dir, "configs")
    if os.path.exists(src_configs):
        print(f"[*] Dang sao chep cau hinh configs sang: {dst_configs}")
        os.makedirs(dst_configs, exist_ok=True)
        for item in os.listdir(src_configs):
            if item.endswith(('.json', '.yaml', '.yml')):
                s = os.path.join(src_configs, item)
                d = os.path.join(dst_configs, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)

    # 4. Sao chep audio_alerts/ sang dist/RailwaySafety/audio_alerts/
    src_audio = os.path.join(base_dir, "audio_alerts")
    dst_audio = os.path.join(target_dir, "audio_alerts")
    if os.path.exists(src_audio):
        print(f"[*] Dang sao chep audio_alerts sang: {dst_audio}")
        if os.path.exists(dst_audio):
            shutil.rmtree(dst_audio)
        shutil.copytree(
            src_audio,
            dst_audio,
            ignore=shutil.ignore_patterns("*.py", "*.pyc", "__pycache__", ".git*")
        )

    # 5. Sao chep ffmpeg.exe sang dist/RailwaySafety/ de ho tro chuyen doi video chuan web H.264
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
        print(f"[*] Dang sao chep ffmpeg.exe sang: {dst_ffmpeg}")
        shutil.copy2(src_ffmpeg, dst_ffmpeg)
    else:
        print("[!] CANH BAO: Khong tim thay ffmpeg.exe de sao chep vao goi phan phoi!")

    # Loai bo ban sao ffmpeg trung lap ben trong _internal de tiet kiem ~84MB
    internal_dir = os.path.join(target_dir, "_internal")
    if os.path.exists(internal_dir):
        for r, d, fs in os.walk(internal_dir):
            for f in fs:
                if "ffmpeg" in f.lower() and f.lower().endswith(".exe"):
                    dup_p = os.path.join(r, f)
                    try:
                        os.remove(dup_p)
                        print(f"[*] Da loai bo ban sao ffmpeg trung lap trong _internal: {dup_p}")
                    except Exception:
                        pass

    # 6. Tao cau truc thu muc data/ ngang hang voi RailwaySafety.exe (khong sao chep video cu de tiet kiem dung luong)
    dst_data = os.path.join(target_dir, "data")
    dst_recordings = os.path.join(dst_data, "recordings")
    dst_logs = os.path.join(dst_data, "intrusion_logs")
    os.makedirs(dst_recordings, exist_ok=True)
    os.makedirs(dst_logs, exist_ok=True)
    print(f"[*] Da tao thu muc data ngang hang tai: {dst_data}")

    # 7. Tao file Khoi_Dong.bat giu lai console de hien thi thong tin va tranh tu dong thoat
    bat_content = """@echo off
title Railway Safety Surveillance System v1.0.0
cd /d "%~dp0"
echo ========================================================
echo    HE THONG GIAM SAT AN TOAN DUONG SAT (v1.0.0)
echo ========================================================
echo Dang khoi chay he thong...
RailwaySafety.exe
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================================
    echo [THONG BAO] Chuong trinh da dung lai voi ma: %ERRORLEVEL%
    echo ========================================================
    pause
)
"""
    bat_path = os.path.join(target_dir, "Khoi_Dong.bat")
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)

    # 8. Tao file README_HUONG_DAN.txt cho nguoi dung cuoi
    readme_content = """========================================================================
   HE THONG GIAM SAT AN TOAN DUONG SAT VA CANH BAO VI PHAM (V1.0.0)
========================================================================

1. CACH KHOI DONG:
   - Click dup vao file "Khoi_Dong.bat" hoac "RailwaySafety.exe".
   - May chu se tu dong khoi chay va mo trinh duyet tai: http://127.0.0.1:5000/
   - Cuaso Console duoc giu lai de theo doi trang thai va log he thong.

2. XEM VIDEO VA LICH SU VI PHAM TRUC TIEP TREN MAY:
   - Thu muc "data" nam ngay canh file "RailwaySafety.exe".
   - Video vi pham: nam tai "data/recordings/{ngay-thang-nam}/"
     -> Ban co the mo truc tiep bang Windows Media Player / VLC / Movies & TV.
   - Nhat ky su kien vi pham: nam tai "data/intrusion_logs/Lich_su_{ngay-thang-nam}.txt".

3. XEM TREN GIAO DIEN WEB:
   - Truy cap muc "Lich su" tren thanh menu trai de xem video va danh sach truc tiep tren web.

4. CAU HINH:
   - Cac thiet lap vung ROI, nguong tin cay, va camera duoc luu tai thu muc "configs/".
========================================================================
"""
    readme_path = os.path.join(target_dir, "README_HUONG_DAN.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("\n" + "=" * 70)
    print("   DONG GOI THANH CONG! SAN PHAM PHAN PHOI SAN SANG TAI:")
    print(f"   {target_dir}")
    print("=" * 70)

if __name__ == '__main__':
    build()
