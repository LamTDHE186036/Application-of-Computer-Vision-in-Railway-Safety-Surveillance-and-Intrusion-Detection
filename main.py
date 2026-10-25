import os
import sys
import socket
import multiprocessing
import webbrowser
from threading import Timer
from app import create_app

app = create_app()

def find_available_port(start_port=5000, max_port=5100):
    """
    Tự động tìm kiếm cổng trống:
    Ưu tiên cổng 5000, nếu bị chiếm thì tự động dò lên 5001, 5002...
    """
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    # Nếu toàn bộ dải đều bận, để hệ điều hành cấp cổng ngẫu nhiên đang trống
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))
        return s.getsockname()[1]

def open_browser(port):
    webbrowser.open_new(f'http://127.0.0.1:{port}/')

if __name__ == '__main__':
    # Hỗ trợ multiprocessing trên Windows khi đóng gói bằng PyInstaller
    multiprocessing.freeze_support()

    port = find_available_port(start_port=5000)

    print("=" * 65)
    print("  HE THONG GIAM SAT AN TOAN DUONG SAT (RAILWAY SAFETY) - V1.0.0")
    print(f"  May chu dang khoi chay tai: http://127.0.0.1:{port}")
    print("  Thu muc luu video & log: 'data' (nam ngay canh chuong trinh)")
    print("=" * 65)

    Timer(1.5, open_browser, args=[port]).start()
    app.run(host='0.0.0.0', port=port, debug=False)
