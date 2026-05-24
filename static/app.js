/* ============================================================
   Sign-O-Text - Client-Side Webcam Capture and Prediction Display
   ============================================================ */

(function () {
    'use strict';

    // -----------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------
    const CONFIG = {
        captureInterval: 100,   // ms between frame captures (~10 fps)
        jpegQuality: 0.6,       // JPEG quality 0-1 (lower = smaller payload)
        maxWidth: 640,          // max capture width
        maxHeight: 480,         // max capture height
        maxHistory: 15,         // max history entries
        noHandDelay: 2000,      // ms before showing "no hand" overlay
    };

    // -----------------------------------------------------------------
    // State
    // -----------------------------------------------------------------
    const state = {
        sessionId: generateSessionId(),
        isRunning: false,
        captureTimer: null,
        noHandTimer: null,
        lastPrediction: null,
        history: [],
    };

    // -----------------------------------------------------------------
    // DOM References
    // -----------------------------------------------------------------
    const dom = {};

    function cacheDom() {
        dom.video = document.getElementById('webcam');
        dom.canvas = document.getElementById('canvas');
        dom.statusDot = document.getElementById('statusDot');
        dom.statusText = document.getElementById('statusText');
        dom.bufferFill = document.getElementById('bufferFill');
        dom.bufferCount = document.getElementById('bufferCount');
        dom.bufferCapacity = document.getElementById('bufferCapacity');
        dom.predictionEmpty = document.getElementById('predictionEmpty');
        dom.predictionResult = document.getElementById('predictionResult');
        dom.predictionCard = document.getElementById('predictionCard');
        dom.predictionSign = document.getElementById('predictionSign');
        dom.confidenceFill = document.getElementById('confidenceFill');
        dom.confidenceText = document.getElementById('confidenceText');
        dom.probabilities = document.getElementById('probabilities');
        dom.history = document.getElementById('history');
        dom.noHandOverlay = document.getElementById('noHandOverlay');
        dom.resetBtn = document.getElementById('resetBtn');
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------
    function generateSessionId() {
        return 'sox-' + Date.now() + '-' + Math.random().toString(36).slice(2, 10);
    }

    function setStatus(text, type) {
        dom.statusText.textContent = text;
        dom.statusDot.className = 'status-dot' + (type ? ' ' + type : '');
    }

    // -----------------------------------------------------------------
    // Webcam Initialization
    // -----------------------------------------------------------------
    async function initWebcam() {
        try {
            setStatus('Requesting camera...');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: CONFIG.maxWidth },
                    height: { ideal: CONFIG.maxHeight },
                    facingMode: 'user',
                },
                audio: false,
            });
            dom.video.srcObject = stream;
            await dom.video.play();

            // Wait for video to have actual dimensions
            await new Promise((resolve) => {
                dom.video.onloadedmetadata = () => {
                    dom.canvas.width = Math.min(dom.video.videoWidth, CONFIG.maxWidth);
                    dom.canvas.height = Math.min(dom.video.videoHeight, CONFIG.maxHeight);
                    resolve();
                };
                // Fallback if already loaded
                if (dom.video.videoWidth > 0) {
                    dom.canvas.width = Math.min(dom.video.videoWidth, CONFIG.maxWidth);
                    dom.canvas.height = Math.min(dom.video.videoHeight, CONFIG.maxHeight);
                    resolve();
                }
            });

            setStatus('Camera ready', 'connected');
            return true;
        } catch (err) {
            console.error('Webcam error:', err);
            setStatus('Camera error: ' + err.message, 'error');
            dom.predictionEmpty.querySelector('.empty-text').textContent =
                'Camera access denied. Please allow camera permissions.';
            return false;
        }
    }

    // -----------------------------------------------------------------
    // Frame Capture & Send
    // -----------------------------------------------------------------
    function captureFrame() {
        const ctx = dom.canvas.getContext('2d');
        // Do NOT mirror here; mirroring is for UI display only.
        // Server should receive the raw orientation to match training data.
        ctx.drawImage(dom.video, 0, 0, dom.canvas.width, dom.canvas.height);

        // Convert to JPEG base64
        return dom.canvas.toDataURL('image/jpeg', CONFIG.jpegQuality);
    }

    async function sendFrame() {
        if (!state.isRunning) return;
        const imageData = captureFrame();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: state.sessionId,
                    image: imageData,
                }),
            });

            if (!response.ok) {
                throw new Error('Server error: ' + response.status);
            }

            const data = await response.json();
            handleResponse(data);
        } catch (err) {
            console.error('Predict error:', err);
            setStatus('Connection error', 'error');
        }
    }

    // -----------------------------------------------------------------
    // Response Handler
    // -----------------------------------------------------------------
    function handleResponse(data) {
        // Update connection status
        setStatus('Running', 'connected');

        // Update buffer
        const pct = Math.min((data.buffer_size / data.buffer_capacity) * 100, 100);
        dom.bufferFill.style.width = pct + '%';
        dom.bufferCount.textContent = data.buffer_size;

        // No-hand overlay
        if (!data.has_hands) {
            if (!state.noHandTimer) {
                state.noHandTimer = setTimeout(() => {
                    dom.noHandOverlay.classList.add('visible');
                }, CONFIG.noHandDelay);
            }
        } else {
            if (state.noHandTimer) {
                clearTimeout(state.noHandTimer);
                state.noHandTimer = null;
            }
            dom.noHandOverlay.classList.remove('visible');
        }

        // Prediction
        if (data.prediction && data.confidence > 0) {
            state.lastPrediction = {
                class: data.prediction,
                confidence: data.confidence,
                timestamp: Date.now(),
            };

            // Show prediction
            dom.predictionEmpty.style.display = 'none';
            dom.predictionResult.style.display = 'block';
            dom.predictionCard.classList.add('has-prediction');
            dom.predictionSign.textContent = data.prediction;

            const confPct = Math.round(data.confidence * 100);
            dom.confidenceFill.style.width = confPct + '%';
            dom.confidenceText.textContent = confPct + '%';

            // Update probabilities
            renderProbabilities(data.all_probs);

            // Add to history
            addToHistory(data.prediction, data.confidence);
        } else {
            // No stable prediction yet
            if (!state.lastPrediction) {
                dom.predictionEmpty.style.display = 'block';
                dom.predictionResult.style.display = 'none';
                dom.predictionCard.classList.remove('has-prediction');
            }
        }
    }

    // -----------------------------------------------------------------
    // UI Renderers
    // -----------------------------------------------------------------
    function renderProbabilities(probs) {
        if (!probs || Object.keys(probs).length === 0) return;

        // Sort by probability descending
        const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        
        dom.probabilities.innerHTML = '';
        sorted.forEach(([cls, prob]) => {
            const pct = Math.round(prob * 100);
            const item = document.createElement('div');
            item.className = 'prob-item';
            item.innerHTML = `
                <div class="prob-name">${cls}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: ${pct}%"></div>
                </div>
                <div class="prob-value">${pct}%</div>
            `;
            dom.probabilities.appendChild(item);
        });
    }

    function addToHistory(prediction, confidence) {
        const now = Date.now();
        
        // Prevent duplicate entries within 2 seconds
        if (state.history.length > 0) {
            const last = state.history[0];
            if (last.class === prediction && (now - last.timestamp) < 2000) {
                return;
            }
        }

        const entry = {
            class: prediction,
            confidence: confidence,
            timestamp: now,
            timeStr: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
        };

        state.history.unshift(entry);
        if (state.history.length > CONFIG.maxHistory) {
            state.history.pop();
        }

        renderHistory();
    }

    function renderHistory() {
        dom.history.innerHTML = '';
        if (state.history.length === 0) {
            dom.history.innerHTML = '<div class="history-empty">No predictions yet</div>';
            return;
        }

        state.history.forEach(entry => {
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <div class="history-main">
                    <span class="history-sign">${entry.class}</span>
                    <span class="history-time">${entry.timeStr}</span>
                </div>
                <div class="history-conf">${Math.round(entry.confidence * 100)}% confidence</div>
            `;
            dom.history.appendChild(item);
        });
    }

    // -----------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------
    async function resetSession() {
        try {
            await fetch('/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: state.sessionId }),
            });
            
            // Clear local state
            state.lastPrediction = null;
            state.history = [];
            dom.predictionEmpty.style.display = 'block';
            dom.predictionResult.style.display = 'none';
            dom.predictionCard.classList.remove('has-prediction');
            dom.probabilities.innerHTML = '<div class="prob-empty">Waiting for predictions...</div>';
            renderHistory();
            
            setStatus('Reset successful', 'connected');
            setTimeout(() => setStatus('Running', 'connected'), 2000);
        } catch (err) {
            console.error('Reset error:', err);
        }
    }

    function startDetection() {
        state.isRunning = true;
        state.captureTimer = setInterval(sendFrame, CONFIG.captureInterval);
    }

    // -----------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------
    async function init() {
        cacheDom();
        
        const cameraOk = await initWebcam();
        if (cameraOk) {
            startDetection();
        }
        
        dom.resetBtn.addEventListener('click', resetSession);
    }

    // Boot
    document.addEventListener('DOMContentLoaded', init);

})();
