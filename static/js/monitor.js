// ===== QUẢN LÝ GIÁM SÁT & CAMERA / VIDEO =====
function setTrainMode(mode) {
    currentTrainMode = mode;
    updateAlarmUI(false);

    fetch('/set_train_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ train_mode: mode })
    }).catch(e => console.warn("Lỗi set_train_mode:", e));
}

function updateAlarmUI(isAlarm) {
    const header = document.getElementById('alarm_header');
    const txt = document.getElementById('alarm_text');
    const btnTrain = document.getElementById('btn_mode_train');
    const btnNoTrain = document.getElementById('btn_mode_notrain');

    if (btnTrain && btnNoTrain) {
        if (currentTrainMode === 'train_approaching') {
            btnTrain.className = 'btn-mode-train active';
            btnNoTrain.className = 'btn-mode-train';
        } else {
            btnTrain.className = 'btn-mode-train';
            btnNoTrain.className = 'btn-mode-train active';
        }
    }

    if (!header || !txt) return;

    header.classList.remove('alarm-off', 'alarm-safe', 'alarm-danger');

    if (currentTrainMode === 'train_approaching') {
        if (isAlarm) {
            header.classList.add('alarm-danger');
            txt.innerText = "Phát hiện vật thể xâm nhập vùng nguy hiểm!";
        } else {
            header.classList.add('alarm-safe');
            txt.innerText = "An toàn!";
        }
    } else {
        if (isAlarm) {
            header.classList.add('alarm-danger');
            txt.innerText = "Vật thể quá lâu trong vùng nguy hiểm!";
        } else {
            header.classList.add('alarm-safe');
            txt.innerText = "An toàn!";
        }
    }
}

// Bắt đầu / dừng giám sát  
function toggleMonitor() {
    if (!isMonitoring) {
        startMonitor();
    } else {
        stopMonitor();
    }
}

function setMode(mode) {
    if (isMonitoring) stopMonitor();
    currentMode = mode;

    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    const btnMap = { 'video': 'btn_vid', 'live': 'btn_live' };
    const activeBtn = document.getElementById(btnMap[mode] || 'btn_vid');
    if (activeBtn) activeBtn.classList.add('active');

    const secVideo = document.getElementById('sec_video');
    const secLive = document.getElementById('sec_live');
    if (secVideo) secVideo.classList.toggle('active', mode === 'video');
    if (secLive) secLive.classList.toggle('active', mode === 'live');

    const mediaTitle = document.getElementById('media_title_text');
    const statusBadge = document.getElementById('media_status_badge');

    if (mode === 'live') {
        isVideoStreamRunning = false;
        fetch('/stop_stream/video').then(() => {
            setTimeout(() => fetch('/delete_video', { method: 'POST' }), 500);
        }).catch(() => {});

        if (mediaTitle) mediaTitle.innerText = "Camera trực tiếp (OBS Virtual Camera)";
        if (statusBadge) {
            statusBadge.innerText = "OBS CAMERA SẴN SÀNG";
            statusBadge.className = "status-badge live";
        }

        const mainImg = document.getElementById('main_preview');
        mainImg.src = "";
        hideMediaPlaceholder(false);
        clearDisplayRoiCanvas();

        startStream('obs', 0);
        if (typeof refreshObsCameraInfo === 'function') {
            refreshObsCameraInfo();
        }
    } else if (mode === 'video') {
        if (mediaTitle) mediaTitle.innerText = "Video giám sát AI";
        if (statusBadge) {
            statusBadge.innerText = "VIDEO SẴN SÀNG";
            statusBadge.className = "status-badge";
        }
        fetch('/stop_stream/obs').catch(() => {});
        fetch('/stop_stream/webcam').catch(() => {});

        const mainImg = document.getElementById('main_preview');
        if (!mainImg.src || mainImg.src === window.location.href) {
            clearDisplayRoiCanvas();
            hideMediaPlaceholder(false);
        } else {
            hideMediaPlaceholder(true);
            setTimeout(drawSavedRoiOnDisplay, 100);
        }
    }
}

function hideMediaPlaceholder(hasContent) {
    const placeholder = document.getElementById('media_placeholder');
    if (placeholder) {
        placeholder.style.display = hasContent ? 'none' : 'flex';
    }
}

// Biến theo dõi trạng thái luồng video đang phát
let isVideoStreamRunning = false;

// Tải video lên
async function uploadVideoPreview(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (isVideoStreamRunning) {
        isVideoStreamRunning = false;
        fetch('/stop_stream/video').catch(() => {});
    }

    const statusText = document.getElementById('vid_status_text');
    if (statusText) statusText.innerText = "Đang xử lý tải video...";

    const formData = new FormData();
    formData.append('video', file);

    try {
        const res = await fetch('/upload_video', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.frame) {
            const mainImg = document.getElementById('main_preview');
            mainImg.src = data.frame;

            const thumb = document.getElementById('thumb_video');
            thumb.src = data.frame;
            thumb.style.display = 'block';

            event.target.closest('.upload-zone').classList.add('has-file');
            if (statusText) statusText.innerText = file.name;

            hideMediaPlaceholder(true);
            setTimeout(drawSavedRoiOnDisplay, 100);
        }
    } catch (err) {
        console.error("Lỗi tải video:", err);
        if (statusText) statusText.innerText = "Lỗi tải video. Hãy thử lại.";
    }
}

// Xóa video đã chọn
function clearMedia(type, event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    const input = document.getElementById('input_' + type);
    if (input) input.value = "";

    const mainImg = document.getElementById('main_preview');
    mainImg.src = "";

    const thumb = document.getElementById('thumb_' + type);
    if (thumb) {
        thumb.src = "";
        thumb.style.display = 'none';
    }

    const zone = document.getElementById('zone_' + type);
    if (zone) zone.classList.remove('has-file');

    const statusText = document.getElementById('vid_status_text');
    if (statusText) statusText.innerText = "Kéo thả hoặc tải video lên";

    updateAlarmUI(false);
    if (isMonitoring) stopMonitor();
    if (type === 'video') isVideoStreamRunning = false;
    clearInterval(pollInterval);
    clearDisplayRoiCanvas();
    hideMediaPlaceholder(false);

    fetch(`/stop_stream/${type}`).then(() => {
        if (type === 'video') {
            setTimeout(() => fetch('/delete_video', { method: 'POST' }), 500);
        }
    }).catch(() => {});
}

// Giám sát
async function startMonitor() {
    if (!savedRoiPoints || savedRoiPoints.length < 3) {
        alert("Vui lòng vẽ và lưu Vùng nguy hiểm (ROI) trước khi bắt đầu giám sát!");
        return;
    }
    if (currentMode === 'video') {
        const videoInput = document.getElementById('input_video');
        if (!videoInput || !videoInput.files[0]) {
            alert("Vui lòng tải video lên trước khi bắt đầu giám sát!");
            return;
        }
        isMonitoring = true;
        updateMonitorButtonUI();
        if (!isVideoStreamRunning) {
            isVideoStreamRunning = true;
            startStream('video', 1);
        } else {
            // Luồng video đã chạy sẵn, chỉ cần bật cờ AI để nhận diện ngay tại thời điểm hiện tại
            fetch('/set_ai/video?ai=1').catch(() => {});
        }
    } else if (currentMode === 'live') {
        isMonitoring = true;
        updateMonitorButtonUI();
        fetch('/set_ai/obs?ai=1').catch(() => {});
    }
}

function startStream(type, ai) {
    const conf = currentConfValue;
    fetch(`/start_stream/${type}?conf=${conf}&ai=${ai}`).then(() => {
        const mainImg = document.getElementById('main_preview');
        mainImg.src = `/video_feed/${type}?t=${new Date().getTime()}`;
        hideMediaPlaceholder(true);

        setTimeout(drawSavedRoiOnDisplay, 200);

        clearInterval(pollInterval);
        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/get_data/${type}`);
                if (res.ok) {
                    const data = await res.json();
                    const logBox = document.getElementById('events_log');
                    if (logBox && data.text) {
                        logBox.innerHTML = data.text;
                    }

                    if (isMonitoring) {
                        updateAlarmUI(data.alarm);
                    } else {
                        updateAlarmUI(false);
                    }
                }
            } catch (e) {
                console.warn("Lỗi đọc dữ liệu luồng:", e);
            }
        }, 500);
    }).catch(e => console.error("Lỗi start_stream:", e));
}

function stopMonitor() {
    isMonitoring = false;
    updateMonitorButtonUI();

    const streamType = currentMode === 'live' ? 'obs' : 'video';
    // Không dừng luồng video (không gọi stop_stream) để hình ảnh vẫn chạy liên tục; chỉ tắt AI
    fetch(`/set_ai/${streamType}?ai=0`).catch(() => {});

    updateAlarmUI(false);

    const badge = document.getElementById('media_status_badge');
    if (badge) {
        badge.innerText = currentMode === 'video' ? "VIDEO SẴN SÀNG" : "OBS CAMERA SẴN SÀNG";
        badge.className = "status-badge";
    }

    setTimeout(async () => {
        try {
            const res = await fetch('/get_events');
            if (res.ok) {
                const data = await res.json();
                const logBox = document.getElementById('events_log');
                if (logBox && data.text) {
                    logBox.innerHTML = data.text;
                }
            }
        } catch (e) {}
    }, 500);

    setTimeout(drawSavedRoiOnDisplay, 100);
}

function updateMonitorButtonUI() {
    const btn = document.getElementById('btn_monitor_toggle');
    const txt = document.getElementById('monitor_text');
    const icon = document.getElementById('monitor_icon');
    const badge = document.getElementById('media_status_badge');

    if (isMonitoring) {
        btn.classList.remove('btn-monitor-start');
        btn.classList.add('btn-monitor-stop');
        txt.innerText = "Dừng giám sát";
        icon.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12"></rect></svg>`;
        if (badge) {
            badge.innerText = "ĐANG GIÁM SÁT AI";
            badge.className = "status-badge live";
        }
    } else {
        btn.classList.remove('btn-monitor-stop');
        btn.classList.add('btn-monitor-start');
        txt.innerText = "Bắt đầu giám sát";
        icon.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
        if (badge) {
            badge.innerText = currentMode === 'video' ? "VIDEO SẴN SÀNG" : "OBS CAMERA SẴN SÀNG";
            badge.className = "status-badge";
        }
    }
    if (typeof updateHistoryLiveStatus === 'function') {
        updateHistoryLiveStatus();
    }
}
