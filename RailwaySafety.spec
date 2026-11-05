# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs
import sys
import os

block_cipher = None

# Thu thập tài nguyên giao diện web tĩnh (configs và audio_alerts được quản lý ngoài thư mục app bởi build_release.py để tiện tùy chỉnh và tránh trùng lặp)
datas = [
    ('templates', 'templates'),
    ('static', 'static'),
]
datas += collect_data_files('ultralytics')
datas += collect_data_files('openvino')

# Thu thập thư viện động C++/DLLs của OpenVINO (ffmpeg.exe được quản lý ngoài thư mục app bởi build_release.py, tránh đóng gói nhân đôi ~88MB)
binaries = []
binaries += collect_dynamic_libs('openvino')

# Thu thập các module ẩn (dynamic imports của YOLO & OpenVINO)
hiddenimports = [
    'cv2',
    'torch',
    'torchvision',
    'openvino',
    'openvino.runtime',
    'ultralytics',
    'ultralytics.models.yolo',
    'ultralytics.nn.modules',
]
hiddenimports += collect_submodules('ultralytics')
hiddenimports += collect_submodules('openvino')

# Loại trừ các thư viện nặng không dùng đến để tối ưu kích thước gói build (~450MB tiết kiệm)
excludes = [
    # Công cụ phát triển & testing
    'tkinter',
    'IPython',
    'jupyter',
    'notebook',
    'tensorboard',
    'pytest',
    'pip',
    'setuptools',
    'wheel',
    # Thư viện phân tích dữ liệu không sử dụng (Tiết kiệm ~360MB)
    'polars',
    '_polars_runtime_32',
    'pyarrow',
    'llvmlite',
    'numba',
    'onnxruntime',
    # Tránh nhân đôi ffmpeg.exe (~84MB) vì ffmpeg.exe đã được quản lý ngoài thư mục app
    'imageio_ffmpeg',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Lọc bỏ triệt để file ffmpeg bị đóng gói nhân đôi trong _internal (tiết kiệm ~84MB)
a.binaries = [x for x in a.binaries if 'imageio_ffmpeg' not in str(x[0]).lower() and not (str(x[0]).lower().endswith('.exe') and 'ffmpeg' in str(x[0]).lower())]
a.datas = [x for x in a.datas if 'imageio_ffmpeg' not in str(x[0]).lower()]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RailwaySafety',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Giữ lại cửa sổ Console cho bản v1.0.0 để theo dõi log & bắt lỗi
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='RailwaySafety',
)
