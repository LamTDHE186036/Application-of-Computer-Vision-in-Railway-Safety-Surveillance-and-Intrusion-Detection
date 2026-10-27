// ===== TOÀN CỤC HỆ THỐNG / GLOBAL STATE =====
let currentMode = 'video';
let currentObsCam = 'auto';
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

let lastDateHyphen = '';

// Khởi tạo đồng hồ Topbar & Tự động phát hiện chuyển ngày (00:00:00)
function updateClock() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('vi-VN');
    const dd = String(now.getDate()).padStart(2, '0');
    const mm = String(now.getMonth() + 1).padStart(2, '0');
    const yyyy = now.getFullYear();
    const dateSlash = `${dd}/${mm}/${yyyy}`;
    const dateHyphen = `${dd}-${mm}-${yyyy}`;

    const topbarTime = document.getElementById('topbar_time');
    const topbarDate = document.getElementById('topbar_date');
    if (topbarTime) topbarTime.innerText = timeStr;
    if (topbarDate) topbarDate.innerText = dateSlash;

    // Tự động chuyển ngày cho khung sự kiện (kể cả khi camera đang dừng)
    if (lastDateHyphen && lastDateHyphen !== dateHyphen) {
        lastDateHyphen = dateHyphen;
        const dateLabel = document.getElementById('current_date_label');
        if (dateLabel) dateLabel.innerText = dateHyphen;

        fetch('/get_events')
            .then(r => r.json())
            .then(evtData => {
                if (evtData && evtData.text !== undefined) {
                    const el = document.getElementById('events_log');
                    if (el) el.innerHTML = evtData.text;
                }
            })
            .catch(() => {});
    } else if (!lastDateHyphen) {
        lastDateHyphen = dateHyphen;
        const dateLabel = document.getElementById('current_date_label');
        if (dateLabel) dateLabel.innerText = dateHyphen;
    }
}

async function refreshObsCameraInfo() {
    try {
        const res = await fetch('/get_cameras');
        if (res.ok) {
            const data = await res.json();
            const tag = document.getElementById('obs_device_tag');
            if (tag) {
                const obsCam = data.cameras.find(c => c.is_obs);
                if (obsCam) {
                    tag.innerText = `${obsCam.name} (Port ${obsCam.index})`;
                    tag.style.background = '#ecfdf5';
                    tag.style.color = '#059669';
                    tag.style.borderColor = '#a7f3d0';
                } else {
                    tag.innerText = `Port ${data.active_index} (Tự động)`;
                }
            }
        }
    } catch (e) {
        console.warn("Lỗi refreshObsCameraInfo:", e);
    }
}

window.addEventListener('DOMContentLoaded', async () => {
    setInterval(updateClock, 1000);
    updateClock();

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
            if (config.obs_cam !== undefined) {
                currentObsCam = config.obs_cam;
            }
            updateAlarmUI(false);
        }
    } catch (err) {
        console.warn("Không kết nối được /get_config:", err);
    }
    refreshObsCameraInfo();
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
