// ===== CHỨC NĂNG TRA CỨU LỊCH SỬ & PHÁT VIDEO VI PHẠM =====
let currentMainView = 'monitor';
let historyEventsData = [];
let filteredHistoryEvents = [];
let selectedHistoryDate = '';
let currentClassFilter = 'all';
let selectedEventIndex = -1;
let currentVideoSpeed = 1.0;
let currentVideoUrl = '';

function switchMainView(viewName) {
    currentMainView = viewName;
    const btnMonitor = document.getElementById('nav_btn_monitor');
    const btnHistory = document.getElementById('nav_btn_history');
    const viewMonitor = document.getElementById('view_monitor');
    const viewHistory = document.getElementById('view_history');

    if (viewName === 'history') {
        if (btnMonitor) btnMonitor.classList.remove('active');
        if (btnHistory) btnHistory.classList.add('active');
        if (viewMonitor) viewMonitor.style.display = 'none';
        if (viewHistory) {
            viewHistory.style.display = 'flex';
            viewHistory.classList.add('active');
        }
        updateHistoryLiveStatus();
        initHistoryView();
    } else {
        if (btnHistory) btnHistory.classList.remove('active');
        if (btnMonitor) btnMonitor.classList.add('active');
        if (viewHistory) {
            viewHistory.style.display = 'none';
            viewHistory.classList.remove('active');
        }
        if (viewMonitor) {
            viewMonitor.style.display = 'grid';
            viewMonitor.classList.add('active');
        }
        // Dừng video lịch sử nếu đang phát để không lẫn âm thanh
        const videoEl = document.getElementById('history_video_element');
        if (videoEl && !videoEl.paused) {
            videoEl.pause();
        }
        // Đồng bộ vẽ lại ROI trên màn hình phân tích nếu cần
        setTimeout(drawSavedRoiOnDisplay, 100);
    }
}

function updateHistoryLiveStatus() {
    const badge = document.getElementById('history_bg_monitor_badge');
    const text = document.getElementById('history_bg_monitor_text');
    if (!badge || !text) return;

    if (isMonitoring) {
        badge.className = 'live-bg-badge';
        text.innerText = currentMode === 'live' ? 'Camera đang giám sát nền (OBS)' : 'Video đang giám sát nền (AI)';
    } else {
        badge.className = 'live-bg-badge idle';
        text.innerText = 'Camera đang tạm dừng (Chờ bắt đầu)';
    }
}

async function initHistoryView() {
    if (!selectedHistoryDate) {
        const today = new Date();
        const dd = String(today.getDate()).padStart(2, '0');
        const mm = String(today.getMonth() + 1).padStart(2, '0');
        const yyyy = today.getFullYear();
        
        selectedHistoryDate = `${dd}-${mm}-${yyyy}`;
        const dateInput = document.getElementById('history_date_input');
        if (dateInput) {
            dateInput.value = `${yyyy}-${mm}-${dd}`;
        }
    }
    await loadHistoryEvents(selectedHistoryDate);
}

function onHistoryDatePicked(val) {
    if (!val) return;
    const parts = val.split('-');
    if (parts.length === 3) {
        selectedHistoryDate = `${parts[2]}-${parts[1]}-${parts[0]}`;
        loadHistoryEvents(selectedHistoryDate);
    }
}

function pickQuickDate(type) {
    const date = new Date();
    if (type === 'yesterday') {
        date.setDate(date.getDate() - 1);
    }
    const dd = String(date.getDate()).padStart(2, '0');
    const mm = String(date.getMonth() + 1).padStart(2, '0');
    const yyyy = date.getFullYear();

    selectedHistoryDate = `${dd}-${mm}-${yyyy}`;
    const dateInput = document.getElementById('history_date_input');
    if (dateInput) {
        dateInput.value = `${yyyy}-${mm}-${dd}`;
    }
    loadHistoryEvents(selectedHistoryDate);
}

function shiftHistoryDate(delta) {
    let baseDate = new Date();
    if (selectedHistoryDate) {
        const parts = selectedHistoryDate.split('-');
        if (parts.length === 3) {
            baseDate = new Date(parseInt(parts[2]), parseInt(parts[1]) - 1, parseInt(parts[0]));
        }
    }
    baseDate.setDate(baseDate.getDate() + delta);
    const dd = String(baseDate.getDate()).padStart(2, '0');
    const mm = String(baseDate.getMonth() + 1).padStart(2, '0');
    const yyyy = baseDate.getFullYear();

    selectedHistoryDate = `${dd}-${mm}-${yyyy}`;
    const dateInput = document.getElementById('history_date_input');
    if (dateInput) {
        dateInput.value = `${yyyy}-${mm}-${dd}`;
    }
    loadHistoryEvents(selectedHistoryDate);
}

function filterHistoryClass(cls) {
    currentClassFilter = cls;
    document.querySelectorAll('.chip-filter').forEach(chip => {
        chip.classList.toggle('active', chip.getAttribute('data-class') === cls);
    });
    applyHistoryFilter();
}

function applyHistoryFilter() {
    if (currentClassFilter === 'all') {
        filteredHistoryEvents = [...historyEventsData];
    } else {
        filteredHistoryEvents = historyEventsData.filter(ev => {
            const name = (ev.object_name || '').toLowerCase();
            const filter = currentClassFilter.toLowerCase();
            return name.includes(filter) || (filter === 'người' && name.includes('nguoi'));
        });
    }
    renderHistoryEvents();
}

async function reloadHistoryEvents() {
    await loadHistoryEvents(selectedHistoryDate);
}

async function loadHistoryEvents(dateStr) {
    const statText = document.getElementById('history_stat_text');
    const countBadge = document.getElementById('history_count_badge');
    if (statText) statText.innerText = "Đang tải dữ liệu...";

    try {
        const res = await fetch(`/api/history/events?date=${encodeURIComponent(dateStr)}`);
        if (res.ok) {
            const data = await res.json();
            historyEventsData = data.events || [];
            const totalVideos = (data.videos || []).length;

            if (statText) {
                statText.innerText = `${historyEventsData.length} vi phạm · ${totalVideos} video lưu`;
            }
            if (countBadge) {
                countBadge.innerText = historyEventsData.length;
            }
            applyHistoryFilter();
        } else {
            if (statText) statText.innerText = "Không thể tải dữ liệu";
        }
    } catch (err) {
        console.error("Lỗi tải lịch sử sự kiện:", err);
        if (statText) statText.innerText = "Lỗi kết nối máy chủ";
    }
}

function getObjectIcon(name) {
    const n = (name || '').toLowerCase();
    if (n.includes('người') || n.includes('person')) return '🚶';
    if (n.includes('ô tô') || n.includes('car')) return '🚗';
    if (n.includes('máy') || n.includes('đạp') || n.includes('motorcycle')) return '🏍️';
    if (n.includes('tải') || n.includes('buýt') || n.includes('truck') || n.includes('bus')) return '🚌';
    return '⚠️';
}

function renderHistoryEvents() {
    const container = document.getElementById('history_events_container');
    if (!container) return;

    if (filteredHistoryEvents.length === 0) {
        container.innerHTML = `
            <div class="history-empty-box">
                <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <span>Không tìm thấy sự kiện vi phạm nào cho ngày này.</span>
            </div>
        `;
        return;
    }

    let html = '';
    filteredHistoryEvents.forEach((ev, idx) => {
        const icon = getObjectIcon(ev.object_name);
        const hasVid = !!ev.video_file;
        const activeCls = selectedEventIndex === idx ? 'active' : '';
        const vidCls = hasVid ? 'has-video' : '';
        const timeRange = ev.end_time && ev.end_time !== ev.start_time ? `${ev.start_time} - ${ev.end_time}` : ev.start_time;

        html += `
            <div class="history-event-card ${activeCls} ${vidCls}" onclick="playHistoryEvent(${idx})">
                <div class="event-card-desc">
                    <b>${ev.object_name}</b> ${ev.desc} - ID: ${ev.track_id}
                </div>
                <div class="event-card-bottom">
                    ${hasVid ? 
                        `<span class="btn-play-pill">
                            <svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                            Xem video
                        </span>` : 
                        `<span class="no-video-pill">Chưa có video</span>`
                    }
                    <span class="event-time-badge">${timeRange}</span>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function playHistoryEvent(idx) {
    selectedEventIndex = idx;
    renderHistoryEvents();

    const ev = filteredHistoryEvents[idx];
    if (!ev) return;

    const playerTitle = document.getElementById('history_player_title');
    const playerBadge = document.getElementById('history_player_badge');
    const placeholder = document.getElementById('history_player_placeholder');
    const videoEl = document.getElementById('history_video_element');
    const streamImg = document.getElementById('history_stream_fallback');
    const metaEvent = document.getElementById('history_meta_event');
    const metaFile = document.getElementById('history_meta_file');
    const slider = document.getElementById('history_timeline_slider');
    const curEl = document.getElementById('history_time_current');
    const totalEl = document.getElementById('history_time_total');

    const timeRange = ev.end_time && ev.end_time !== ev.start_time ? `${ev.start_time} - ${ev.end_time}` : ev.start_time;
    if (playerTitle) playerTitle.innerText = `${ev.object_name} vi phạm [${timeRange}]`;
    if (metaEvent) metaEvent.innerText = `Sự kiện: ${ev.object_name} (ID ${ev.track_id}) - ${ev.desc} lúc ${timeRange}`;

    stopFallbackTimer();
    isUserSeeking = false;

    if (ev.video_file && ev.video_url) {
        currentVideoUrl = ev.video_url;
        if (placeholder) placeholder.style.display = 'none';
        if (streamImg) {
            streamImg.style.display = 'none';
            streamImg.src = '';
        }

        const knownDur = ev.video_duration || 0;
        if (slider) {
            slider.value = 0;
            slider.max = knownDur > 0 ? knownDur : 100;
        }
        if (curEl) curEl.innerText = "00:00";
        if (totalEl) totalEl.innerText = formatTimeMMSS(knownDur);

        if (videoEl) {
            videoEl.style.display = 'block';
            videoEl.src = ev.video_url;
            videoEl.playbackRate = currentVideoSpeed;
            videoEl.load();
            videoEl.play().catch(err => {
                console.warn("Autoplay bị chặn hoặc người dùng chưa tương tác:", err);
            });
        }

        if (playerBadge) {
            playerBadge.innerText = "ĐANG PHÁT VIDEO";
            playerBadge.className = "status-badge live";
        }
        if (metaFile) {
            metaFile.innerText = `Tập tin: data/recordings/${ev.date}/${ev.video_file}`;
        }
    } else {
        currentVideoUrl = '';
        if (slider) slider.value = 0;
        if (curEl) curEl.innerText = "00:00";
        if (totalEl) totalEl.innerText = "00:00";

        if (videoEl) {
            videoEl.pause();
            videoEl.style.display = 'none';
            videoEl.src = '';
        }
        if (streamImg) {
            streamImg.style.display = 'none';
            streamImg.src = '';
        }
        if (placeholder) {
            placeholder.style.display = 'flex';
            placeholder.querySelector('h3').innerText = "Không tìm thấy video cho sự kiện này";
            placeholder.querySelector('p').innerText = "Sự kiện được ghi lại trong nhật ký nhưng chưa có clip tương ứng trong thư mục data/recordings.";
        }
        if (playerBadge) {
            playerBadge.innerText = "KHÔNG CÓ VIDEO";
            playerBadge.className = "status-badge";
        }
        if (metaFile) metaFile.innerText = "Không có tệp video";
    }
}

// ===== XỬ LÝ CHẾ ĐỘ PHÁT STREAM DỰ PHÒNG (MJPEG STREAM FALLBACK) =====
let fallbackTimer = null;
let currentFallbackSec = 0;
let fallbackIsPlaying = false;

function onVideoPlaybackError(e) {
    const videoEl = document.getElementById('history_video_element');
    // Chỉ kích hoạt fallback khi videoEl thực sự đang cố phát một URL hợp lệ và bị lỗi codec/nguồn
    if (!videoEl || !videoEl.src || videoEl.src === window.location.href || !videoEl.error) {
        return;
    }

    const ev = filteredHistoryEvents[selectedEventIndex];
    if (!ev || !ev.video_file) return;

    console.warn("Trình duyệt không thể giải mã video HTML5, tự động chuyển sang luồng stream mjpeg:", videoEl.error);
    const streamImg = document.getElementById('history_stream_fallback');
    const playerBadge = document.getElementById('history_player_badge');
    const slider = document.getElementById('history_timeline_slider');
    const totalEl = document.getElementById('history_time_total');

    videoEl.style.display = 'none';
    videoEl.pause();

    const duration = ev.video_duration || 30;
    if (slider) slider.max = duration;
    if (totalEl) totalEl.innerText = formatTimeMMSS(duration);

    if (streamImg) {
        streamImg.style.display = 'block';
        const startSec = slider ? parseFloat(slider.value) || 0 : 0;
        startFallbackPlayback(startSec);
    }
    if (playerBadge) {
        playerBadge.innerText = "PHÁT DẠNG LUỒNG ẢNH";
        playerBadge.className = "status-badge live";
    }
}

function startFallbackPlayback(startSec) {
    const ev = filteredHistoryEvents[selectedEventIndex];
    if (!ev || !ev.video_file) return;

    currentFallbackSec = startSec;
    fallbackIsPlaying = true;
    const streamImg = document.getElementById('history_stream_fallback');
    if (streamImg) {
        streamImg.src = `/history_video_stream?date=${encodeURIComponent(ev.date)}&file=${encodeURIComponent(ev.video_file)}&start_sec=${startSec}&t=${Date.now()}`;
    }

    const playIcon = document.getElementById('icon_history_play');
    if (playIcon) {
        playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    }

    stopFallbackTimer();
    const maxDur = ev.video_duration || 60;
    fallbackTimer = setInterval(() => {
        if (!isUserSeeking && fallbackIsPlaying) {
            currentFallbackSec += 1;
            if (currentFallbackSec >= maxDur) {
                currentFallbackSec = maxDur;
                stopFallbackTimer();
                fallbackIsPlaying = false;
                if (playIcon) playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
            }
            const slider = document.getElementById('history_timeline_slider');
            const curEl = document.getElementById('history_time_current');
            if (slider) slider.value = currentFallbackSec;
            if (curEl) curEl.innerText = formatTimeMMSS(currentFallbackSec);
        }
    }, 1000);
}

function stopFallbackTimer() {
    if (fallbackTimer) {
        clearInterval(fallbackTimer);
        fallbackTimer = null;
    }
    fallbackIsPlaying = false;
}

function setVideoSpeed(spd) {
    currentVideoSpeed = spd;
    document.querySelectorAll('.btn-speed').forEach(btn => {
        btn.classList.toggle('active', btn.innerText === spd + 'x');
    });
    const videoEl = document.getElementById('history_video_element');
    if (videoEl) {
        videoEl.playbackRate = spd;
    }
}

function downloadCurrentVideo() {
    if (!currentVideoUrl) {
        alert("Vui lòng chọn một sự kiện có video vi phạm trước khi tải!");
        return;
    }
    const a = document.createElement('a');
    a.href = currentVideoUrl;
    a.download = currentVideoUrl.split('/').pop() || 'violation_video.mp4';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// ===== XỬ LÝ THANH TUA TIMELINE VIDEO LỊCH SỬ (PHÚT:GIÂY) =====
let isUserSeeking = false;

function formatTimeMMSS(seconds) {
    if (isNaN(seconds) || seconds < 0 || !isFinite(seconds)) return "00:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
}

function onHistoryTimelineInput(val) {
    isUserSeeking = true;
    const curEl = document.getElementById('history_time_current');
    if (curEl) curEl.innerText = formatTimeMMSS(parseFloat(val));
}

function onHistoryTimelineChange(val) {
    const targetSec = parseFloat(val);
    const videoEl = document.getElementById('history_video_element');
    const streamImg = document.getElementById('history_stream_fallback');
    const curEl = document.getElementById('history_time_current');

    if (curEl) curEl.innerText = formatTimeMMSS(targetSec);

    if (videoEl && videoEl.style.display !== 'none' && videoEl.src) {
        videoEl.currentTime = targetSec;
        if (videoEl.paused) {
            videoEl.play().catch(() => {});
        }
    } else if (streamImg && streamImg.style.display !== 'none') {
        startFallbackPlayback(targetSec);
    }
    isUserSeeking = false;
}

function seekRelative(deltaSeconds) {
    const slider = document.getElementById('history_timeline_slider');
    if (!slider) return;
    const curVal = parseFloat(slider.value) || 0;
    const maxVal = parseFloat(slider.max) || 100;
    const nextVal = Math.max(0, Math.min(maxVal, curVal + deltaSeconds));
    slider.value = nextVal;
    onHistoryTimelineChange(nextVal);
}

function toggleHistoryVideoPlay() {
    const videoEl = document.getElementById('history_video_element');
    const streamImg = document.getElementById('history_stream_fallback');

    if (videoEl && videoEl.style.display !== 'none' && videoEl.src) {
        if (videoEl.paused) {
            videoEl.play();
        } else {
            videoEl.pause();
        }
    } else if (streamImg && streamImg.style.display !== 'none') {
        const slider = document.getElementById('history_timeline_slider');
        const icon = document.getElementById('icon_history_play');
        if (fallbackIsPlaying) {
            fallbackIsPlaying = false;
            stopFallbackTimer();
            streamImg.src = '';
            if (icon) icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
        } else {
            const startSec = slider ? parseFloat(slider.value) : 0;
            startFallbackPlayback(startSec);
        }
    }
}

function initTimelineListeners() {
    const videoEl = document.getElementById('history_video_element');
    if (!videoEl) return;

    videoEl.addEventListener('error', onVideoPlaybackError);

    videoEl.addEventListener('loadedmetadata', () => {
        const slider = document.getElementById('history_timeline_slider');
        const totalEl = document.getElementById('history_time_total');
        if (slider && videoEl.duration && !isNaN(videoEl.duration)) {
            slider.max = videoEl.duration;
        }
        if (totalEl && videoEl.duration && !isNaN(videoEl.duration)) {
            totalEl.innerText = formatTimeMMSS(videoEl.duration);
        }
    });

    videoEl.addEventListener('timeupdate', () => {
        if (!isUserSeeking) {
            const slider = document.getElementById('history_timeline_slider');
            const curEl = document.getElementById('history_time_current');
            if (slider && !isNaN(videoEl.currentTime)) {
                slider.value = videoEl.currentTime;
            }
            if (curEl && !isNaN(videoEl.currentTime)) {
                curEl.innerText = formatTimeMMSS(videoEl.currentTime);
            }
        }
    });

    videoEl.addEventListener('play', () => {
        const icon = document.getElementById('icon_history_play');
        if (icon) {
            icon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
        }
    });

    videoEl.addEventListener('pause', () => {
        const icon = document.getElementById('icon_history_play');
        if (icon) {
            icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
        }
    });

    videoEl.addEventListener('ended', () => {
        const icon = document.getElementById('icon_history_play');
        if (icon) {
            icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
        }
    });
}

if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', initTimelineListeners);
} else {
    initTimelineListeners();
}

