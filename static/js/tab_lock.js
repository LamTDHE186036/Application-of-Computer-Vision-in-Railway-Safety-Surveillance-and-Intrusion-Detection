// ===== KHÓA ĐƠN TAB (SINGLE-TAB LOCK) =====
// Đảm bảo chỉ có DUY NHẤT 1 tab trình duyệt được phép kết nối và xử lý video/AI tại một thời điểm.

const TAB_LOCK_STORAGE_KEY = 'railway_active_tab_id';
const TAB_LOCK_HEARTBEAT_KEY = 'railway_active_tab_hb';
const TAB_CHANNEL_NAME = 'railway_single_tab_channel';
const HEARTBEAT_INTERVAL_MS = 1500;
const HEARTBEAT_TIMEOUT_MS = 4000;

const currentTabId = 'tab_' + Math.random().toString(36).substring(2, 9) + '_' + Date.now();
let isThisTabBlocked = false;
let tabHeartbeatTimer = null;
let broadcastChannel = null;

try {
    if (typeof BroadcastChannel !== 'undefined') {
        broadcastChannel = new BroadcastChannel(TAB_CHANNEL_NAME);
        broadcastChannel.onmessage = (event) => {
            const data = event.data || {};
            if (data.action === 'claim' && data.tabId !== currentTabId) {
                // Tab khác đã chiếm quyền điều khiển -> Khóa tab này
                blockCurrentTab();
            } else if (data.action === 'tab_closed' && data.tabId !== currentTabId) {
                // Tab chính đã đóng -> Cho phép tab này tự động kích hoạt
                if (isThisTabBlocked) {
                    claimTabControl(true);
                }
            }
        };
    }
} catch (e) {
    console.warn("BroadcastChannel không khả dụng, sử dụng localStorage fallback:", e);
}

function sendBroadcast(action) {
    if (broadcastChannel) {
        try {
            broadcastChannel.postMessage({ action: action, tabId: currentTabId, time: Date.now() });
        } catch (e) {}
    }
}

function updateHeartbeat() {
    try {
        localStorage.setItem(TAB_LOCK_STORAGE_KEY, currentTabId);
        localStorage.setItem(TAB_LOCK_HEARTBEAT_KEY, String(Date.now()));
    } catch (e) {}
}

function startHeartbeat() {
    stopHeartbeat();
    updateHeartbeat();
    tabHeartbeatTimer = setInterval(updateHeartbeat, HEARTBEAT_INTERVAL_MS);
}

function stopHeartbeat() {
    if (tabHeartbeatTimer) {
        clearInterval(tabHeartbeatTimer);
        tabHeartbeatTimer = null;
    }
}

function checkTabLockOnStartup() {
    try {
        const activeTabId = localStorage.getItem(TAB_LOCK_STORAGE_KEY);
        const lastHbStr = localStorage.getItem(TAB_LOCK_HEARTBEAT_KEY);
        const lastHb = lastHbStr ? parseInt(lastHbStr, 10) : 0;
        const now = Date.now();

        // Nếu đã có tab khác đang hoạt động và còn gửi nhịp tim trong vòng 4 giây
        if (activeTabId && activeTabId !== currentTabId && (now - lastHb < HEARTBEAT_TIMEOUT_MS)) {
            blockCurrentTab();
            return false;
        }
    } catch (e) {}

    // Chiếm quyền tab chủ sở hữu
    unblockCurrentTab();
    startHeartbeat();
    sendBroadcast('claim');
    return true;
}

function blockCurrentTab() {
    isThisTabBlocked = true;
    stopHeartbeat();

    // Dừng toàn bộ âm thanh và luồng video nếu đang chạy
    if (typeof forceStopAlarmAudio === 'function') {
        try { forceStopAlarmAudio(); } catch (e) {}
    }
    if (typeof stopAlarmAudioGracefully === 'function') {
        try { stopAlarmAudioGracefully(); } catch (e) {}
    }

    const overlay = document.getElementById('duplicate_tab_overlay');
    if (overlay) {
        overlay.style.display = 'flex';
    }
}

function unblockCurrentTab() {
    isThisTabBlocked = false;
    const overlay = document.getElementById('duplicate_tab_overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function claimTabControl(auto = false) {
    unblockCurrentTab();
    startHeartbeat();
    sendBroadcast('claim');

    // Làm mới luồng stream nếu cần
    if (typeof setMode === 'function' && typeof currentMode !== 'undefined') {
        try { setMode(currentMode); } catch (e) {}
    }
}

function isTabBlocked() {
    return isThisTabBlocked;
}

// Lắng nghe sự kiện trước khi đóng tab để nhượng quyền
window.addEventListener('beforeunload', () => {
    try {
        const activeTabId = localStorage.getItem(TAB_LOCK_STORAGE_KEY);
        if (activeTabId === currentTabId) {
            localStorage.removeItem(TAB_LOCK_STORAGE_KEY);
            localStorage.removeItem(TAB_LOCK_HEARTBEAT_KEY);
            sendBroadcast('tab_closed');
        }
    } catch (e) {}
});

// Khởi chạy kiểm tra ngay khi nạp file
checkTabLockOnStartup();
window.addEventListener('DOMContentLoaded', () => {
    if (isThisTabBlocked) {
        const overlay = document.getElementById('duplicate_tab_overlay');
        if (overlay) overlay.style.display = 'flex';
    }
});
