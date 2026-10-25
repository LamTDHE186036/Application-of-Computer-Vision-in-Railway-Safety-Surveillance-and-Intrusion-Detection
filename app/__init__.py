import os
import atexit
from flask import Flask
from app import config
from app.routes import register_blueprints

def cleanup_on_shutdown():
    temp_path = os.path.join(config.BASE_DIR, "temp_video.mp4")
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

atexit.register(cleanup_on_shutdown)

def create_app():
    """
    Application factory khởi tạo ứng dụng Flask với đường dẫn tĩnh và giao diện chuẩn
    """
    template_dir = os.path.join(config.BASE_DIR, "templates")
    static_dir = os.path.join(config.BASE_DIR, "static")

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir
    )

    # Đăng ký các route phân theo domain (stream, history, settings)
    register_blueprints(app)

    return app

app = create_app()
