# app/routes/__init__.py
from app.routes.stream import stream_bp
from app.routes.history import history_bp
from app.routes.settings import settings_bp

def register_blueprints(app):
    """Đăng ký tất cả các blueprints phân theo domain vào ứng dụng Flask"""
    app.register_blueprint(stream_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(settings_bp)

__all__ = ["stream_bp", "history_bp", "settings_bp", "register_blueprints"]
