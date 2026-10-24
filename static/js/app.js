// ===== TOÀN CỤC HỆ THỐNG / GLOBAL STATE =====
let currentMode = 'video';
let alertEnabled = false;
let isMonitoring = false;
let pollInterval;
let savedRoiPoints = [];
let roiPoints = [];
let isDrawingPolygon = false;
let currentConfValue = 0.5;
let currentClassesList = ["person", "car", "truck-bus", "motorcycle"];
let currentTrainMode = 'train_approaching';
let currentDwellValue = 5.0;

// Khởi tạo đồng hồ Topbar
function updateClock() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('vi-VN');
    const dateStr = String(now.getDate()).padStart(2, '0') + '/' + String(now.getMonth() + 1).padStart(2, '0') + '/' + now.getFullYear();
    const topbarTime = document.getElementById('topbar_time');
    const topbarDate = document.getElementById('topbar_date');
    if (topbarTime) topbarTime.innerText = timeStr;
    if (topbarDate) topbarDate.innerText = dateStr;
}

window.addEventListener('DOMContentLoaded', async () => {
    setInterval(updateClock, 1000);
    updateClock();

    const today = new Date();
    const dateStr = String(today.getDate()).padStart(2, '0') + '-' + String(today.getMonth() + 1).padStart(2, '0') + '-' + today.getFullYear();
    const dateLabel = document.getElementById('current_date_label');
    if (dateLabel) dateLabel.innerText = dateStr;

    try {
        const res = await fetch('/get_config');
        if (res.ok) {
            const config = await res.json();
            if (config.polygon && config.polygon.length >= 3) {
                savedRoiPoints = config.polygon;
                updateRoiPointCount();
            }
            if (config.conf) {
                currentConfValue = parseFloat(config.conf);
            }
            if (config.classes && Array.isArray(config.classes)) {
                currentClassesList = config.classes;
            }
            if (config.dwell_time !== undefined) {
                currentDwellValue = parseFloat(config.dwell_time);
            }
            if (config.train_mode) {
                currentTrainMode = config.train_mode;
            }
            updateAlarmUI(false);
        }
    } catch (err) {
        console.warn("Không kết nối được /get_config:", err);
    }
    updateRoiStatusNotice();

    try {
        const evtRes = await fetch('/get_events');
        if (evtRes.ok) {
            const evtData = await evtRes.json();
            if (evtData && evtData.text) {
                document.getElementById('events_log').innerHTML = evtData.text;
            }
        }
    } catch (err) {
        console.warn("Không kết nối được /get_events:", err);
    }

    setMode('video');
});
