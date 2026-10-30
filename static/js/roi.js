// ===== KHỞI TẠO CANVAS VÙNG NGUY HIỂM (ROI) =====
const canvas = document.getElementById('roi_canvas');
const ctx = canvas ? canvas.getContext('2d') : null;
const displayCanvas = document.getElementById('roi_display_canvas');
const displayCtx = displayCanvas ? displayCanvas.getContext('2d') : null;

// ===== XỬ LÝ VÙNG NGUY HIỂM (ROI) =====
function resizeCanvas() {
    const mainImg = document.getElementById('main_preview');
    const camContainer = document.getElementById('cam_container');
    if (mainImg && mainImg.src && mainImg.clientWidth > 0 && mainImg.clientHeight > 0) {
        const imgRect = mainImg.getBoundingClientRect();
        const containerRect = camContainer.getBoundingClientRect();
        const w = imgRect.width;
        const h = imgRect.height;
        const left = imgRect.left - containerRect.left;
        const top = imgRect.top - containerRect.top;
        [canvas, displayCanvas].forEach(c => {
            if (c) {
                c.width = w;
                c.height = h;
                c.style.width = w + 'px';
                c.style.height = h + 'px';
                c.style.left = left + 'px';
                c.style.top = top + 'px';
                c.style.right = 'auto';
                c.style.bottom = 'auto';
            }
        });
    } else if (camContainer) {
        const w = camContainer.clientWidth;
        const h = camContainer.clientHeight;
        if (w > 0 && h > 0) {
            [canvas, displayCanvas].forEach(c => {
                if (c) {
                    c.width = w;
                    c.height = h;
                    c.style.width = '100%';
                    c.style.height = '100%';
                    c.style.left = '0px';
                    c.style.top = '0px';
                }
            });
        }
    }
}

window.addEventListener('resize', () => {
    resizeCanvas();
    if (isDrawingPolygon) {
        drawPolygon();
    } else {
        drawSavedRoiOnDisplay();
    }
});

function updateRoiPointCount() {
    const label = document.getElementById('roi_point_count');
    if (label) {
        const count = isDrawingPolygon ? roiPoints.length : savedRoiPoints.length;
        label.innerText = `${count} điểm đã chọn`;
    }
    updateRoiStatusNotice();
}

function updateRoiStatusNotice() {
    const textEl = document.getElementById('roi_status_text');
    const hintEl = document.getElementById('roi_status_hint');
    if (!textEl) return;

    const hasRoi = Array.isArray(savedRoiPoints) && savedRoiPoints.length >= 3;
    if (hasRoi) {
        textEl.innerText = "Đã có vùng nguy hiểm";
        if (hintEl) hintEl.classList.add('has-roi');
    } else {
        textEl.innerText = "Chưa có vùng nguy hiểm";
        if (hintEl) hintEl.classList.remove('has-roi');
    }
}

function enableROIDraw() {
    canvas.style.display = 'block';
    resizeCanvas();
    roiPoints = [...savedRoiPoints];
    isDrawingPolygon = true;
    updateRoiPointCount();
    drawPolygon();

    const badge = document.getElementById('media_status_badge');
    if (badge) {
        badge.innerText = "CHẾ ĐỘ VẼ ROI";
        badge.className = "status-badge drawing";
    }
}

function undoLastPoint() {
    if (isDrawingPolygon && roiPoints.length > 0) {
        roiPoints.pop();
        updateRoiPointCount();
        drawPolygon();
    }
}

function clearROI() {
    roiPoints = [];
    savedRoiPoints = [];
    isDrawingPolygon = false;
    canvas.style.display = 'none';
    clearDisplayRoiCanvas();
    updateRoiPointCount();

    if (isMonitoring) {
        stopMonitor();
    }

    const badge = document.getElementById('media_status_badge');
    if (badge) {
        badge.innerText = isMonitoring ? "ĐANG GIÁM SÁT AI" : (currentMode === 'live' ? "OBS CAMERA SẴN SÀNG" : "VIDEO SẴN SÀNG");
        badge.className = isMonitoring ? "status-badge live" : "status-badge";
    }

    fetch('/set_roi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: false, polygon: [] })
    }).then(() => {
        alert("Đã xóa vùng nguy hiểm thành công!");
    }).catch(e => console.warn("Lỗi xóa ROI:", e));
}

function clearDisplayRoiCanvas() {
    displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
}

canvas.addEventListener('click', (e) => {
    if (!isDrawingPolygon) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    roiPoints.push({ x: x / canvas.width, y: y / canvas.height });
    updateRoiPointCount();
    drawPolygon();
});

function drawPolygon() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (roiPoints.length === 0) return;

    ctx.beginPath();
    ctx.moveTo(roiPoints[0].x * canvas.width, roiPoints[0].y * canvas.height);
    for (let i = 1; i < roiPoints.length; i++) {
        ctx.lineTo(roiPoints[i].x * canvas.width, roiPoints[i].y * canvas.height);
    }

    if (roiPoints.length >= 3) {
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 224, 68, 0.25)';
        ctx.fill();
    }

    ctx.strokeStyle = '#ffe044';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    roiPoints.forEach((p, idx) => {
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#e2ab00';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

function drawSavedRoiOnDisplay() {
    const mainImg = document.getElementById('main_preview');
    if (!mainImg.src || mainImg.src === window.location.href || mainImg.clientWidth === 0) {
        clearDisplayRoiCanvas();
        return;
    }

    resizeCanvas();
    clearDisplayRoiCanvas();
    if (savedRoiPoints.length < 3) return;

    displayCtx.beginPath();
    displayCtx.moveTo(savedRoiPoints[0].x * displayCanvas.width, savedRoiPoints[0].y * displayCanvas.height);
    for (let i = 1; i < savedRoiPoints.length; i++) {
        displayCtx.lineTo(savedRoiPoints[i].x * displayCanvas.width, savedRoiPoints[i].y * displayCanvas.height);
    }
    displayCtx.closePath();
    displayCtx.fillStyle = 'rgba(255, 224, 68, 0.2)';
    displayCtx.fill();
    displayCtx.strokeStyle = '#0d828a';
    displayCtx.lineWidth = 2.5;
    displayCtx.stroke();
}

function saveROI() {
    if (!isDrawingPolygon) return;

    if (roiPoints.length > 0 && roiPoints.length < 3) {
        alert("Vui lòng chấm ít nhất 3 điểm để tạo thành một khu vực khép kín!");
        return;
    }
    isDrawingPolygon = false;
    canvas.style.display = 'none';

    savedRoiPoints = [...roiPoints];
    updateRoiPointCount();
    drawSavedRoiOnDisplay();

    const badge = document.getElementById('media_status_badge');
    if (badge) {
        badge.innerText = isMonitoring ? "ĐANG GIÁM SÁT AI" : (currentMode === 'live' ? "OBS CAMERA SẴN SÀNG" : "VIDEO SẴN SÀNG");
        badge.className = isMonitoring ? "status-badge live" : "status-badge";
    }

    const isEnabled = savedRoiPoints.length >= 3;
    fetch('/set_roi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: isEnabled, polygon: savedRoiPoints })
    }).then(() => {
        if (isEnabled) alert("Đã lưu và áp dụng vùng nguy hiểm thành công!");
    }).catch(e => console.warn("Lỗi lưu ROI:", e));
}
