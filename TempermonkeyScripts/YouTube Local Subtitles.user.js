// ==UserScript==
// @name         YouTube Local Subtitles
// @namespace    http://tampermonkey.net/
// @version      0.5
// @description  Play local .srt subtitles for YouTube videos served via a local HTTP server (default http://localhost:8765/)
// @author       king.ho
// @match        https://www.youtube.com/*
// @match        https://m.youtube.com/*
// @match        https://music.youtube.com/*
// @grant        GM_xmlhttpRequest
// @connect      localhost
// @connect      127.0.0.1
// @run-at       document-idle
// ==/UserScript==

(function () {
    'use strict';

    const LOG_PREFIX = '[YT-LocalSRT]';
    function log(...args) {
        console.log(LOG_PREFIX, ...args);
    }
    function warn(...args) {
        console.warn(LOG_PREFIX, ...args);
    }

    const config = {
        srtBaseUrl: 'http://localhost:8765/',
        languageVariants: [
            { suffix: '.zh-Hant', label: 'zh-Hant' },
            { suffix: '.zh-TW', label: 'zh-TW' },
            { suffix: '.zh', label: 'zh' },
            { suffix: '', label: '' },
            { suffix: '.en', label: 'en' }
        ],
        fontSize: 22,
        bottomOffset: 80,
        urlPollIntervalMs: 1000
    };

    const state = {
        buttonEl: null,
        subtitleEl: null,
        cues: [],
        videoEl: null,
        onTimeUpdate: null,
        currentVideoId: null,
        loadedLanguage: '',
        active: false
    };

    function extractVideoId() {
        const url = new URL(location.href);
        if (url.pathname === '/watch') return url.searchParams.get('v');
        const m = url.pathname.match(/\/(?:embed|shorts)\/([^/?]+)/);
        return m ? m[1] : null;
    }

    function init() {
        log('script loaded, url=', location.href);
        observeUrlChanges();
        evaluateCurrentPage();
    }

    function observeUrlChanges() {
        document.addEventListener('yt-navigate-finish', evaluateCurrentPage);
        let lastUrl = location.href;
        setInterval(() => {
            if (location.href !== lastUrl) {
                lastUrl = location.href;
                evaluateCurrentPage();
            }
        }, config.urlPollIntervalMs);
    }

    function evaluateCurrentPage() {
        const videoId = extractVideoId();
        if (videoId === state.currentVideoId) return;
        teardown();
        state.currentVideoId = videoId;
        if (!videoId) {
            log('no videoId on this page, url=', location.href);
            return;
        }
        log('evaluating videoId=', videoId);
        loadSrt(videoId).then(result => {
            if (state.currentVideoId !== videoId) {
                log('videoId changed during fetch, abort', videoId);
                return;
            }
            if (!result) {
                warn('no SRT file found for', videoId, '— tried variants:', config.languageVariants.map(v => v.suffix + '.srt').join(', '));
                return;
            }
            state.cues = parseSrt(result.text);
            if (state.cues.length === 0) {
                warn('SRT loaded but 0 cues parsed, language=', result.label || 'default');
                return;
            }
            state.loadedLanguage = result.label;
            log('loaded', state.cues.length, 'cues, language=', result.label || 'default');
            showButton();
        });
    }

    async function loadSrt(videoId) {
        for (const variant of config.languageVariants) {
            const text = await fetchSrt(videoId, variant.suffix);
            if (text) return { text, label: variant.label };
        }
        return null;
    }

    function fetchSrt(videoId, suffix) {
        const url = config.srtBaseUrl + encodeURIComponent(videoId) + suffix + '.srt';
        return new Promise(resolve => {
            GM_xmlhttpRequest({
                method: 'GET',
                url,
                onload: res => {
                    const bytes = res.responseText ? res.responseText.length : 0;
                    const ok = res.status === 200 || (res.status === 0 && bytes > 0);
                    log('fetch', url, '-> status=', res.status, 'bytes=', bytes, ok ? 'OK' : 'MISS');
                    resolve(ok ? res.responseText : null);
                },
                onerror: err => {
                    warn('fetch error', url, err);
                    resolve(null);
                },
                ontimeout: () => {
                    warn('fetch timeout', url);
                    resolve(null);
                }
            });
        });
    }

    function parseSrt(text) {
        const cues = [];
        const blocks = text.replace(/\r/g, '').split(/\n\n+/);
        for (const block of blocks) {
            const lines = block.split('\n').filter(l => l.length > 0);
            if (lines.length < 2) continue;
            const timeIdx = lines.findIndex(l => l.includes('-->'));
            if (timeIdx === -1) continue;
            const [startStr, endStr] = lines[timeIdx].split('-->').map(s => s.trim());
            const start = parseTimestamp(startStr);
            const end = parseTimestamp(endStr);
            if (end <= start) continue;
            const cueText = lines.slice(timeIdx + 1).join('\n');
            cues.push({ start, end, text: cueText });
        }
        return cues;
    }

    function parseTimestamp(s) {
        const m = s.match(/(\d+):(\d+):(\d+)[,.](\d+)/);
        if (!m) return 0;
        return parseInt(m[1], 10) * 3600
            + parseInt(m[2], 10) * 60
            + parseInt(m[3], 10)
            + parseInt(m[4], 10) / 1000;
    }

    function buildButtonLabel(active) {
        const base = active ? '關閉字幕' : '本地字幕';
        return state.loadedLanguage ? `${base} (${state.loadedLanguage})` : base;
    }

    function showButton() {
        const btn = document.createElement('button');
        btn.className = '__local_srt_button';
        btn.textContent = buildButtonLabel(false);
        Object.assign(btn.style, {
            position: 'fixed',
            top: '70px',
            right: '20px',
            zIndex: '2147483647',
            padding: '8px 14px',
            background: 'rgba(33, 150, 243, 0.95)',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: 'bold',
            cursor: 'pointer',
            boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
            fontFamily: '-apple-system, "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif'
        });
        btn.addEventListener('click', toggleSubtitles);
        document.body.appendChild(btn);
        state.buttonEl = btn;
        log('button rendered, label=', btn.textContent);
    }

    function toggleSubtitles() {
        if (state.active) {
            stopSubtitles();
            applyButtonStyle({ background: 'rgba(33, 150, 243, 0.95)', text: buildButtonLabel(false) });
        } else {
            startSubtitles();
            if (state.active) {
                applyButtonStyle({ background: 'rgba(244, 67, 54, 0.95)', text: buildButtonLabel(true) });
            }
        }
    }

    function applyButtonStyle({ background, text }) {
        if (!state.buttonEl) return;
        state.buttonEl.style.background = background;
        state.buttonEl.textContent = text;
    }

    function startSubtitles() {
        const video = document.querySelector('video');
        const player = document.querySelector('#movie_player')
            || document.querySelector('.html5-video-player');
        if (!video || !player) {
            warn('cannot find video or player element, video=', !!video, 'player=', !!player);
            return;
        }
        state.videoEl = video;
        state.subtitleEl = createSubtitleElement();
        player.appendChild(state.subtitleEl);
        state.onTimeUpdate = () => updateSubtitleText(video.currentTime);
        video.addEventListener('timeupdate', state.onTimeUpdate);
        updateSubtitleText(video.currentTime);
        state.active = true;
    }

    function stopSubtitles() {
        if (state.videoEl && state.onTimeUpdate) {
            state.videoEl.removeEventListener('timeupdate', state.onTimeUpdate);
        }
        if (state.subtitleEl) {
            state.subtitleEl.remove();
        }
        state.subtitleEl = null;
        state.onTimeUpdate = null;
        state.videoEl = null;
        state.active = false;
    }

    function createSubtitleElement() {
        const el = document.createElement('div');
        el.className = '__local_srt_subtitle';
        Object.assign(el.style, {
            position: 'absolute',
            left: '50%',
            bottom: config.bottomOffset + 'px',
            transform: 'translateX(-50%)',
            maxWidth: '80%',
            padding: '6px 14px',
            background: 'rgba(0, 0, 0, 0.75)',
            color: '#fff',
            fontSize: config.fontSize + 'px',
            fontFamily: '-apple-system, "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif',
            textAlign: 'center',
            whiteSpace: 'pre-wrap',
            borderRadius: '4px',
            pointerEvents: 'none',
            zIndex: '60',
            textShadow: '1px 1px 2px rgba(0, 0, 0, 0.8)',
            display: 'none'
        });
        return el;
    }

    function updateSubtitleText(currentTime) {
        if (!state.subtitleEl) return;
        const cue = state.cues.find(c => currentTime >= c.start && currentTime <= c.end);
        if (cue) {
            state.subtitleEl.textContent = cue.text;
            state.subtitleEl.style.display = 'block';
        } else {
            state.subtitleEl.style.display = 'none';
        }
    }

    function teardown() {
        stopSubtitles();
        if (state.buttonEl) {
            state.buttonEl.remove();
            state.buttonEl = null;
        }
        state.cues = [];
        state.loadedLanguage = '';
    }

    init();
})();
