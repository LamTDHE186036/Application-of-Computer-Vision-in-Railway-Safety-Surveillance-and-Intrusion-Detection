# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs
import sys
import os

block_cipher = None

# Thu thập tài nguyên dữ liệu tĩnh
datas = [
    ('templates', 'templates'),
    ('static', 'static'),
    ('configs', 'configs'),
    ('audio_alerts', 'audio_alerts'),
]
datas += collect_data_files('ultralytics')
datas += collect_data_files('openvino')

# Thu thập thư viện động C++/DLLs
binaries = []
binaries += collect_dynamic_libs('openvino')

# Thu thập các module ẩn (dynamic imports của YOLO & OpenVINO)
hiddenimports = [
    'engineio.async_drivers.threading',
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

# Loại trừ các thư viện nặng không dùng đến để tối ưu kích thước gói build
excludes = [
    'tkinter',
    'IPython',
    'jupyter',
    'notebook',
    'tensorboard',
    'pytest',
    'pip',
    'setuptools',
    'wheel',
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
