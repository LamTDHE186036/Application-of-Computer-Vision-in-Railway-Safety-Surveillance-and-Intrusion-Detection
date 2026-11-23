async function loadCamerasList() {
    const camSelect = document.getElementById('modal_obs_cam_select');
    if (!camSelect) return;

    try {
        const res = await fetch('/get_cameras?t=' + new Date().getTime());
        if (!res.ok) throw new Error("Mã phản hồi: " + res.status);
        const data = await res.json();

        camSelect.innerHTML = '';

        const optAuto = document.createElement('option');
        optAuto.value = 'auto';
        optAuto.text = 'Tự động chọn (Ưu tiên OBS Virtual Camera / Webcam)';
        camSelect.appendChild(optAuto);

        if (data && data.cameras && Array.isArray(data.cameras) && data.cameras.length > 0) {
            data.cameras.forEach(cam => {
                const opt = document.createElement('option');
                opt.value = String(cam.index);
                let label = `Camera [Cổng ${cam.index}]: ${cam.name}`;
                if (cam.is_obs) {
                    label += ' ⭐ (OBS Virtual Camera)';
                }
                opt.text = label;
                camSelect.appendChild(opt);
            });
        }

        const currentVal = String(currentObsCam || (data ? data.configured : 'auto') || 'auto');
        camSelect.value = currentVal;
    } catch (e) {
        console.warn("Lỗi tải danh sách camera:", e);
    }
}

// ===== MODAL CÀI ĐẶT THAM SỐ AI =====
function openSettingsModal() {
    const modal = document.getElementById('settings_modal');
    if (!modal) return;

    // Đồng bộ giá trị Confidence
    const confSlider = document.getElementById('modal_conf_slider');
    const confInput = document.getElementById('modal_conf_input');
    if (confSlider) confSlider.value = currentConfValue;
    if (confInput) confInput.value = currentConfValue;

    // Đồng bộ giá trị Dwell Time (thời gian dừng đỗ cho phép)
    const dwellSlider = document.getElementById('modal_dwell_slider');
    const dwellInput = document.getElementById('modal_dwell_input');
    if (dwellSlider) dwellSlider.value = currentDwellValue;
    if (dwellInput) dwellInput.value = currentDwellValue;

    // Đồng bộ các checkbox lớp nhận dạng (4 lớp sử dụng)
    const classKeys = ['person', 'car', 'motorcycle', 'truck-bus'];
    const activeClasses = new Set(currentClassesList || []);

    classKeys.forEach(k => {
        const cb = document.getElementById('class_' + k);
        const card = document.getElementById('card_' + k);
        if (cb) {
            cb.checked = activeClasses.has(k);
            if (card) card.classList.toggle('checked', cb.checked);
        }
    });

    // Đồng bộ danh sách Camera cho Live Stream
    loadCamerasList();

    modal.classList.add('active');
}

function closeSettingsModal() {
    const modal = document.getElementById('settings_modal');
    if (modal) modal.classList.remove('active');
}

function syncModalConf(val) {
    let num = parseFloat(val);
    if (isNaN(num)) return;
    if (num < 0.1) num = 0.1;
    if (num > 1.0) num = 1.0;
    document.getElementById('modal_conf_slider').value = num;
    document.getElementById('modal_conf_input').value = num;
}

function syncModalDwell(val) {
    let num = parseFloat(val);
    if (isNaN(num)) return;
    if (num < 1) num = 1;
    if (num > 60) num = 60;
    const dwellSlider = document.getElementById('modal_dwell_slider');
    const dwellInput = document.getElementById('modal_dwell_input');
    if (dwellSlider) dwellSlider.value = num;
    if (dwellInput) dwellInput.value = num;
}

function onClassCheckboxChange(cb) {
    const card = cb.closest('.class-card');
    if (card) card.classList.toggle('checked', cb.checked);
}

function toggleAllClasses(checked) {
    const checkboxes = document.querySelectorAll('.class-checkbox');
    checkboxes.forEach(cb => {
        cb.checked = checked;
        onClassCheckboxChange(cb);
    });
}

async function saveSettingsFromModal() {
    const confVal = parseFloat(document.getElementById('modal_conf_input').value) || 0.5;
    const dwellVal = parseFloat(document.getElementById('modal_dwell_input').value) || 5.0;
    const checkedBoxes = Array.from(document.querySelectorAll('.class-checkbox:checked'));
    const selectedClasses = checkedBoxes.map(cb => cb.value);

    const camSelect = document.getElementById('modal_obs_cam_select');
    const obsCamVal = camSelect ? camSelect.value : 'auto';

    if (selectedClasses.length === 0) {
        if (!confirm("Bạn chưa chọn lớp đối tượng nào! Hệ thống sẽ không phát hiện vi phạm. Bạn có chắc muốn lưu?")) {
            return;
        }
    }

    try {
        const payload = {
            conf: confVal,
            classes: selectedClasses,
            dwell_time: dwellVal,
            obs_cam: obsCamVal
        };

        const res = await fetch('/set_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            currentConfValue = confVal;
            currentClassesList = selectedClasses;
            currentDwellValue = dwellVal;
            currentObsCam = obsCamVal;
            
            if (typeof refreshObsCameraInfo === 'function') {
                refreshObsCameraInfo();
            }

            // Nếu đang xem trực tiếp camera và chưa có luồng thì mới kích hoạt
            if (currentMode === 'live') {
                const mainImg = document.getElementById('main_preview');
                if (!mainImg || !mainImg.src || mainImg.src.indexOf('/video_feed/obs') === -1) {
                    startStream('obs', isMonitoring ? 1 : 0);
                }
            }

            updateAlarmUI(false);
            closeSettingsModal();
            alert("Đã lưu cấu hình AI thành công!");
        } else {
            alert("Không thể lưu cấu hình, vui lòng thử lại!");
        }
    } catch (err) {
        console.error("Lỗi lưu cấu hình:", err);
        alert("Lỗi kết nối khi lưu cấu hình!");
    }
}
