import os
import atexit
from flask import Flask
from app import config
from app.routes import register_blueprints

def cleanup_on_shutdown():
    if os.path.exists(config.TEMP_VIDEO_PATH):
        try:
            os.remove(config.TEMP_VIDEO_PATH)
        except Exception:
            pass

atexit.register(cleanup_on_shutdown)

def create_app():
    """
    Application factory khởi tạo ứng dụng Flask với đường dẫn tĩnh và giao diện chuẩn
    """
    # Tìm template_dir: ưu tiên trong BUNDLE_DIR, sau đó đến BASE_DIR
    template_dir = os.path.join(config.BUNDLE_DIR, "templates")
    if not os.path.exists(template_dir):
        template_dir = os.path.join(config.BASE_DIR, "templates")

    # Tìm static_dir: ưu tiên trong BUNDLE_DIR, sau đó đến BASE_DIR
    static_dir = os.path.join(config.BUNDLE_DIR, "static")
    if not os.path.exists(static_dir):
        static_dir = os.path.join(config.BASE_DIR, "static")

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir
    )

    # Đăng ký các route phân theo domain (stream, history, settings)
    register_blueprints(app)

    return app
