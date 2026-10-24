from flask import Flask
from core.routes import register_routes
import webbrowser
from threading import Timer

app = Flask(__name__)

# Đăng ký toàn bộ URL/Routes vào ứng dụng
register_routes(app)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1.5, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=False)