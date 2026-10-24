// ===== MODAL CÀI ĐẶT THAM SỐ AI =====
function openSettingsModal() {
    const modal = document.getElementById('settings_modal');
    if (!modal) return;

    // Đồng bộ giá trị Confidence
    document.getElementById('modal_conf_slider').value = currentConfValue;
    document.getElementById('modal_conf_input').value = currentConfValue;

    // Đồng bộ giá trị Dwell Time (thời gian dừng đỗ cho phép)
    const dwellSlider = document.getElementById('modal_dwell_slider');
    const dwellInput = document.getElementById('modal_dwell_input');
    if (dwellSlider) dwellSlider.value = currentDwellValue;
    if (dwellInput) dwellInput.value = currentDwellValue;

    // Đồng bộ các checkbox lớp nhận dạng
    const classKeys = ['person', 'car', 'truck_bus', 'motorcycle'];
    const valMap = {
        'person': 'person',
        'car': 'car',
        'truck_bus': 'truck-bus',
        'motorcycle': 'motorcycle'
    };

    classKeys.forEach(k => {
        const val = valMap[k];
        const cb = document.getElementById('class_' + k);
        const card = document.getElementById('card_' + k);
        if (cb) {
            cb.checked = currentClassesList.includes(val);
            if (card) card.classList.toggle('checked', cb.checked);
        }
    });

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

    if (selectedClasses.length === 0) {
        if (!confirm("Bạn chưa chọn lớp đối tượng nào! Hệ thống sẽ không phát hiện vi phạm. Bạn có chắc muốn lưu?")) {
            return;
        }
    }

    try {
        const res = await fetch('/set_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conf: confVal,
                classes: selectedClasses,
                dwell_time: dwellVal
            })
        });

        if (res.ok) {
            const data = await res.json();
            currentConfValue = confVal;
            currentClassesList = selectedClasses;
            currentDwellValue = dwellVal;
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
