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

    # 2. Sao chep models/ sang dist/RailwaySafety/models/
    src_models = os.path.join(base_dir, "models")
    dst_models = os.path.join(target_dir, "models")
    if os.path.exists(src_models):
        print(f"[*] Dang sao chep mo hinh AI sang: {dst_models}")
        if os.path.exists(dst_models):
            shutil.rmtree(dst_models)
        shutil.copytree(src_models, dst_models)

    # 3. Sao chep configs/ sang dist/RailwaySafety/configs/
    src_configs = os.path.join(base_dir, "configs")
    dst_configs = os.path.join(target_dir, "configs")
    if os.path.exists(src_configs):
        print(f"[*] Dang sao chep cau hinh configs sang: {dst_configs}")
        if not os.path.exists(dst_configs):
            shutil.copytree(src_configs, dst_configs)
        else:
            for item in os.listdir(src_configs):
                s = os.path.join(src_configs, item)
                d = os.path.join(dst_configs, item)
                if not os.path.exists(d) and os.path.isfile(s):
                    shutil.copy2(s, d)

    # 4. Tao cau truc thu muc data/ ngang hang voi RailwaySafety.exe
    dst_data = os.path.join(target_dir, "data")
    dst_recordings = os.path.join(dst_data, "recordings")
    dst_logs = os.path.join(dst_data, "intrusion_logs")
    os.makedirs(dst_recordings, exist_ok=True)
    os.makedirs(dst_logs, exist_ok=True)
    print(f"[*] Da tao thu muc data ngang hang tai: {dst_data}")

    # 5. Tao file Khoi_Dong.bat giu lai console de hien thi thong tin va tranh tu dong thoat
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

    # 6. Tao file README_HUONG_DAN.txt cho nguoi dung cuoi
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
