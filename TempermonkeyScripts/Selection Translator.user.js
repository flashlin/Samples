// ==UserScript==
// @name         Selection Translator (Ollama)
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  Select text -> blue dot -> tooltip translation via local Ollama (OpenAI-compatible)
// @author       king.ho
// @match        *://*/*
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// @connect      localhost
// @require      https://cdn.jsdelivr.net/npm/opencc-js@1.3.1/dist/umd/full.js
// @run-at       document-idle
// ==/UserScript==

(function () {
    'use strict';

    const config = {
        endpoint: 'http://127.0.0.1:11434/v1/chat/completions',
        model: 'sun_leaf/HY-MT:1.8b',
        delayMs: 500,
        maxLines: 8,
        fontSize: 14,
        lineHeight: 1.5,
        timeoutMs: 60000,
        systemPrompt: 'You are a translation engine. Translate ALL input text into Traditional Chinese (Taiwan). For mixed-language content, translate every part properly. Preserve line breaks. Output ONLY the translated result, no explanations, no quotes, no language labels.'
    };

    const state = {
        dotEl: null,
        tooltipEl: null,
        timer: null,
        currentText: '',
        selectionRect: null,
        s2tConverter: buildS2tConverter()
    };

    function buildS2tConverter() {
        try {
            if (typeof OpenCC !== 'undefined' && OpenCC.Converter) {
                return OpenCC.Converter({ from: 'cn', to: 'twp' });
            }
        } catch (e) {
            console.warn('[Selection Translator] OpenCC init failed:', e);
        }
        return (text) => text;
    }

    function toTraditional(text) {
        if (!text) return text;
        try {
            return state.s2tConverter(text);
        } catch (e) {
            return text;
        }
    }

    function init() {
        document.addEventListener('mouseup', onMouseUp, true);
        document.addEventListener('mousedown', onMouseDown, true);
        document.addEventListener('scroll', cleanupAll, true);
        window.addEventListener('resize', cleanupAll, true);
    }

    function isInsideOurUi(target) {
        if (state.dotEl && state.dotEl.contains(target)) return true;
        if (state.tooltipEl && state.tooltipEl.contains(target)) return true;
        return false;
    }

    function onMouseDown(e) {
        if (isInsideOurUi(e.target)) return;
        cleanupAll();
    }

    function onMouseUp(e) {
        if (isInsideOurUi(e.target)) return;
        clearTimeout(state.timer);
        const mouse = { x: e.clientX, y: e.clientY };
        state.timer = setTimeout(() => handleSelection(mouse), config.delayMs);
    }

    function handleSelection(mouse) {
        const picked = pickSelection();
        if (!picked) return;
        const targetRect = findNearestRect({ rects: picked.lineRects, mouse });
        cleanupAll();
        state.currentText = picked.text;
        state.selectionRect = picked.boundingRect;
        showDot({ targetRect, mouse });
    }

    function pickSelection() {
        const sel = window.getSelection();
        if (!sel || sel.isCollapsed) return null;
        const text = sel.toString().trim();
        if (!text) return null;
        const range = sel.getRangeAt(0);
        const lineRects = Array.from(range.getClientRects()).filter(r => r.width > 0 && r.height > 0);
        if (lineRects.length === 0) return null;
        return { text, lineRects, boundingRect: range.getBoundingClientRect() };
    }

    function findNearestRect({ rects, mouse }) {
        let best = rects[0];
        let bestDist = Infinity;
        for (const r of rects) {
            const cx = Math.max(r.left, Math.min(mouse.x, r.right));
            const cy = Math.max(r.top, Math.min(mouse.y, r.bottom));
            const dx = mouse.x - cx;
            const dy = mouse.y - cy;
            const d = dx * dx + dy * dy;
            if (d < bestDist) {
                bestDist = d;
                best = r;
            }
        }
        return best;
    }

    function showDot({ targetRect, mouse }) {
        const dot = createDotElement();
        const pos = computeDotPosition({ targetRect, mouse, dotSize: 14 });
        dot.style.left = pos.x + 'px';
        dot.style.top = pos.y + 'px';
        dot.addEventListener('mousedown', stopEvent);
        dot.addEventListener('mouseup', stopEvent);
        dot.addEventListener('click', (e) => {
            stopEvent(e);
            onDotClick();
        });
        document.body.appendChild(dot);
        state.dotEl = dot;
    }

    function createDotElement() {
        const dot = document.createElement('div');
        dot.className = '__ollama_translator_dot';
        Object.assign(dot.style, {
            position: 'fixed',
            width: '14px',
            height: '14px',
            borderRadius: '50%',
            background: '#2196f3',
            boxShadow: '0 2px 6px rgba(0,0,0,0.35)',
            cursor: 'pointer',
            zIndex: '2147483647',
            border: '2px solid white',
            boxSizing: 'content-box'
        });
        return dot;
    }

    function computeDotPosition({ targetRect, mouse, dotSize }) {
        const offset = 6;
        const candidates = [
            { x: targetRect.right + offset, y: targetRect.top - dotSize - offset },
            { x: targetRect.right + offset, y: targetRect.bottom + offset },
            { x: targetRect.left - dotSize - offset, y: targetRect.top - dotSize - offset },
            { x: targetRect.left - dotSize - offset, y: targetRect.bottom + offset },
            { x: targetRect.right + offset, y: targetRect.top + (targetRect.height - dotSize) / 2 },
            { x: targetRect.left - dotSize - offset, y: targetRect.top + (targetRect.height - dotSize) / 2 }
        ];
        let best = candidates[0];
        let bestDist = Infinity;
        for (const c of candidates) {
            const dx = mouse.x - (c.x + dotSize / 2);
            const dy = mouse.y - (c.y + dotSize / 2);
            const d = dx * dx + dy * dy;
            if (d < bestDist) {
                bestDist = d;
                best = c;
            }
        }
        best.x = Math.max(4, Math.min(window.innerWidth - dotSize - 4, best.x));
        best.y = Math.max(4, Math.min(window.innerHeight - dotSize - 4, best.y));
        return best;
    }

    function onDotClick() {
        removeDot();
        const tooltip = createTooltipElement();
        document.body.appendChild(tooltip);
        state.tooltipEl = tooltip;
        setTooltipContent('Translating...');
        translate(state.currentText)
            .then(result => setTooltipContent(toTraditional(result)))
            .catch(err => setTooltipContent('[Error] ' + err.message));
    }

    function createTooltipElement() {
        const wrap = document.createElement('div');
        wrap.className = '__ollama_translator_tooltip';
        const maxH = config.maxLines * config.fontSize * config.lineHeight + 20;
        Object.assign(wrap.style, {
            position: 'fixed',
            background: 'rgba(28, 28, 30, 0.97)',
            color: '#fff',
            padding: '10px 12px',
            borderRadius: '8px',
            fontSize: config.fontSize + 'px',
            lineHeight: String(config.lineHeight),
            boxShadow: '0 6px 24px rgba(0,0,0,0.4)',
            zIndex: '2147483647',
            maxHeight: maxH + 'px',
            overflowY: 'auto',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontFamily: '-apple-system, "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif',
            boxSizing: 'border-box'
        });
        wrap.addEventListener('mousedown', stopEvent);
        wrap.addEventListener('mouseup', stopEvent);
        wrap.addEventListener('click', stopEvent);
        wrap.addEventListener('wheel', (e) => e.stopPropagation(), { passive: true });
        return wrap;
    }

    function setTooltipContent(text) {
        if (!state.tooltipEl) return;
        state.tooltipEl.textContent = text;
        positionTooltip();
    }

    function positionTooltip() {
        const tooltip = state.tooltipEl;
        const base = state.selectionRect;
        if (!tooltip || !base) return;
        const width = Math.max(140, Math.min(base.width, window.innerWidth - 16));
        tooltip.style.width = width + 'px';
        tooltip.style.left = '0px';
        tooltip.style.top = '0px';
        const measuredHeight = tooltip.offsetHeight;
        const spaceBelow = window.innerHeight - base.bottom;
        const placeBelow = spaceBelow >= measuredHeight + 10 || spaceBelow >= base.top;
        const top = placeBelow
            ? Math.min(window.innerHeight - measuredHeight - 4, base.bottom + 6)
            : Math.max(4, base.top - measuredHeight - 6);
        const left = Math.max(8, Math.min(window.innerWidth - width - 8, base.left));
        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    }

    function translate(text) {
        return new Promise((resolve, reject) => {
            GM_xmlhttpRequest({
                method: 'POST',
                url: config.endpoint,
                headers: { 'Content-Type': 'application/json' },
                timeout: config.timeoutMs,
                data: JSON.stringify({
                    model: config.model,
                    messages: [
                        { role: 'system', content: config.systemPrompt },
                        { role: 'user', content: text }
                    ],
                    stream: false,
                    temperature: 0.2
                }),
                onload: (res) => handleTranslateResponse({ res, resolve, reject }),
                onerror: () => reject(new Error('Request failed (is Ollama running?)')),
                ontimeout: () => reject(new Error('Request timeout'))
            });
        });
    }

    function handleTranslateResponse({ res, resolve, reject }) {
        if (res.status < 200 || res.status >= 300) {
            reject(new Error('HTTP ' + res.status + ' ' + (res.responseText || '').slice(0, 200)));
            return;
        }
        try {
            const data = JSON.parse(res.responseText);
            const content = data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
            const trimmed = (content || '').trim();
            if (!trimmed) {
                reject(new Error('Empty response'));
                return;
            }
            resolve(trimmed);
        } catch (err) {
            reject(new Error('Parse error: ' + err.message));
        }
    }

    function stopEvent(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function removeDot() {
        if (state.dotEl) {
            state.dotEl.remove();
            state.dotEl = null;
        }
    }

    function removeTooltip() {
        if (state.tooltipEl) {
            state.tooltipEl.remove();
            state.tooltipEl = null;
        }
    }

    function cleanupAll() {
        clearTimeout(state.timer);
        removeDot();
        removeTooltip();
    }

    init();
})();
