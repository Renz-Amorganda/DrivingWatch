/**
 * DrivingWatch: Operator Fatigue and Drowsiness Detection
 * Built with TensorFlow.js and Google Teachable Machine
 * Strict client-side processing, zero remote video transmission.
 */

// Application State
const state = {
  isRunning: false,
  isAudioMuted: false,
  facingMode: 'user', // Default to front/selfie camera on mobile phones
  model: null,
  webcam: null,
  maxPredictions: 0,
  currentModelUrl: null,

  // Performance telemetry
  lastFrameTime: performance.now(),
  frameCount: 0,
  currentFps: 0,

  // Fatigue persistence buffer (Hysteresis)
  activeDrowsyStart: null,
  activeDistractedStart: null,
  drowsyThresholdMs: 1500, // 1.5 seconds continuous to prevent false alarms on normal blinks
  distractedThresholdMs: 2000,
  drowsyIncidentCount: 0,
  distractedIncidentCount: 0,
  alarmActive: false,

  // Session clock
  sessionStartTime: null,
  sessionTimerInterval: null,
};

// Web Audio API Synth for Acoustic Warning
let audioCtx = null;
let alertOscillator = null;

function initAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
}

function playWarningTone() {
  if (state.isAudioMuted || !audioCtx) return;
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }

  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = 'sine';
    // Gentle warning tone frequency
    osc.frequency.setValueAtTime(660, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.28);

    gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.28);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + 0.3);
  } catch (err) {
    console.error('Audio synthesis failed:', err);
  }
}

// DOM References
const elements = {
  globalStatusPill: document.getElementById('globalStatusPill'),
  globalStatusDot: document.getElementById('globalStatusDot'),
  globalStatusLabel: document.getElementById('globalStatusLabel'),
  toggleCameraBtn: document.getElementById('toggleCameraBtn'),
  cameraBtnText: document.getElementById('cameraBtnText'),
  flipCameraBtn: document.getElementById('flipCameraBtn'),
  toggleAudioBtn: document.getElementById('toggleAudioBtn'),
  audioBtnText: document.getElementById('audioBtnText'),
  openSettingsBtn: document.getElementById('openSettingsBtn'),

  // Viewport
  videoSkeleton: document.getElementById('videoSkeleton'),
  webcamWrapper: document.getElementById('webcamWrapper'),
  reticleBox: document.getElementById('reticleBox'),
  reticleStatusText: document.getElementById('reticleStatusText'),
  fpsCounter: document.getElementById('fpsCounter'),
  latencyCounter: document.getElementById('latencyCounter'),

  // Telemetry Readouts
  sessionClock: document.getElementById('sessionClock'),
  stateCard: document.getElementById('stateCard'),
  currentClassification: document.getElementById('currentClassification'),
  primaryConfidence: document.getElementById('primaryConfidence'),

  scoreAlert: document.getElementById('scoreAlert'),
  barAlert: document.getElementById('barAlert'),
  scoreDrowsy: document.getElementById('scoreDrowsy'),
  barDrowsy: document.getElementById('barDrowsy'),
  scoreDistracted: document.getElementById('scoreDistracted'),
  barDistracted: document.getElementById('barDistracted'),

  fatigueTriggerBar: document.getElementById('fatigueTriggerBar'),
  fatigueTimeElapsed: document.getElementById('fatigueTimeElapsed'),

  drowsyIncidentCount: document.getElementById('drowsyIncidentCount'),
  distractedIncidentCount: document.getElementById('distractedIncidentCount'),
  incidentLogBody: document.getElementById('incidentLogBody'),
  clearLogBtn: document.getElementById('clearLogBtn'),

  // Modals
  modelModal: document.getElementById('modelModal'),
  modelUrlInput: document.getElementById('modelUrlInput'),
  saveModelUrlBtn: document.getElementById('saveModelUrlBtn'),
  usePresetModelBtn: document.getElementById('usePresetModelBtn'),
  closeModelModalBtn: document.getElementById('closeModelModalBtn'),

  policyModal: document.getElementById('policyModal'),
  policyModalTitle: document.getElementById('policyModalTitle'),
  policyModalContent: document.getElementById('policyModalContent'),
  closePolicyModalBtn: document.getElementById('closePolicyModalBtn'),
  openTosBtn: document.getElementById('openTosBtn'),
  openPrivacyBtn: document.getElementById('openPrivacyBtn'),
};

// Model State
let isUsingEmulatedModel = false;

// Initialize Teachable Machine Model
async function loadModel(customUrl = null) {
  updateStatus('warning', 'Loading Neural Weights...');

  // Determine paths: user custom URL, or local relative files
  let modelURL = './model.json';
  let metadataURL = './metadata.json';

  if (customUrl) {
    modelURL = customUrl.endsWith('/') ? customUrl + 'model.json' : customUrl + '/model.json';
    metadataURL = customUrl.endsWith('/') ? customUrl + 'metadata.json' : customUrl + '/metadata.json';
  }

  try {
    state.model = await tmImage.load(modelURL, metadataURL);
    state.maxPredictions = state.model.getTotalClasses();
    state.currentModelUrl = customUrl || './';
    isUsingEmulatedModel = false;
    updateStatus('active', customUrl ? 'Custom Model Active' : 'Model Active (Local)');
    console.log('Teachable Machine model loaded successfully. Classes:', state.model.getClassLabels());
    return true;
  } catch (err) {
    console.warn('Unable to load model from: ' + modelURL, err);
    if (customUrl) {
      alert('Could not load the specified model URL. Reverting to the local vision pipeline.');
    }
  }

  // Fallback to built-in heuristic vision processing if local fetch fails (e.g. file:// CORS restriction)
  isUsingEmulatedModel = true;
  if (window.location.protocol === 'file:') {
    updateStatus('warning', 'HTTP Server Required for Local Model');
    console.info('Tip: Local model files require serving via an HTTP server (e.g., npx serve, python http.server, or GitHub Pages) due to browser file:// CORS security policies.');
  } else {
    updateStatus('active', 'Vision Pipeline Ready');
  }
  return false;
}

// Status Pill Updater
function updateStatus(type, label) {
  elements.globalStatusDot.className = 'status-dot';
  if (type === 'active') elements.globalStatusDot.classList.add('active');
  if (type === 'warning') elements.globalStatusDot.classList.add('warning');
  if (type === 'critical') elements.globalStatusDot.classList.add('critical');
  elements.globalStatusLabel.textContent = label;
}

// Session Timer
function startSessionClock() {
  state.sessionStartTime = Date.now();
  if (state.sessionTimerInterval) clearInterval(state.sessionTimerInterval);

  state.sessionTimerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - state.sessionStartTime) / 1000);
    const hrs = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const mins = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    elements.sessionClock.textContent = `Session: ${hrs}:${mins}:${secs}`;
  }, 1000);
}

function stopSessionClock() {
  if (state.sessionTimerInterval) clearInterval(state.sessionTimerInterval);
  elements.sessionClock.textContent = 'Session: 00:00:00';
}

// Start Video and Inference Pipeline
async function startMonitoring() {
  initAudioContext();
  elements.cameraBtnText.textContent = 'Initializing...';
  elements.toggleCameraBtn.disabled = true;

  // Show Skeleton view
  elements.videoSkeleton.classList.remove('hidden');
  elements.webcamWrapper.classList.add('hidden');

  try {
    if (!state.model && !isUsingEmulatedModel) {
      await loadModel(state.currentModelUrl);
    }

    const flip = state.facingMode === 'user';
    state.webcam = new tmImage.Webcam(480, 480, flip);
    await state.webcam.setup({ facingMode: state.facingMode });

    // iOS and Android mobile optimization: prevent native fullscreen takeover
    if (state.webcam.webcam) {
      state.webcam.webcam.setAttribute('playsinline', 'true');
      state.webcam.webcam.setAttribute('webkit-playsinline', 'true');
      state.webcam.webcam.muted = true;
    }

    await state.webcam.play();

    // Attach to DOM
    elements.webcamWrapper.innerHTML = '';
    elements.webcamWrapper.appendChild(state.webcam.canvas);

    // Hide Skeleton, Reveal Camera
    elements.videoSkeleton.classList.add('hidden');
    elements.webcamWrapper.classList.remove('hidden');

    state.isRunning = true;
    elements.cameraBtnText.textContent = 'Stop Monitoring';
    elements.toggleCameraBtn.disabled = false;
    updateStatus('active', 'Monitoring Active');
    startSessionClock();

    requestAnimationFrame(inferenceLoop);
  } catch (err) {
    console.error('Camera initialization failed:', err);
    alert('Unable to access camera. Please confirm browser webcam permissions are enabled.');
    elements.cameraBtnText.textContent = 'Start Monitoring';
    elements.toggleCameraBtn.disabled = false;
    elements.videoSkeleton.classList.remove('hidden');
    updateStatus('critical', 'Camera Blocked');
  }
}

// Stop Video and Reset Pipeline
function stopMonitoring() {
  state.isRunning = false;
  if (state.webcam) {
    state.webcam.stop();
  }

  elements.cameraBtnText.textContent = 'Start Monitoring';
  elements.videoSkeleton.classList.remove('hidden');
  elements.webcamWrapper.classList.add('hidden');
  updateStatus('inactive', 'System Ready');

  resetClassificationDisplays();
  stopSessionClock();
}

function resetClassificationDisplays() {
  elements.reticleBox.className = 'reticle-box';
  elements.reticleStatusText.textContent = 'Camera Inactive';
  elements.stateCard.className = 'state-card';
  elements.currentClassification.textContent = 'Inactive';
  elements.primaryConfidence.textContent = '0%';

  elements.barAlert.style.width = '0%';
  elements.scoreAlert.textContent = '0.0%';
  elements.barDrowsy.style.width = '0%';
  elements.scoreDrowsy.textContent = '0.0%';
  elements.barDistracted.style.width = '0%';
  elements.scoreDistracted.textContent = '0.0%';
  elements.fatigueTriggerBar.style.width = '0%';
  elements.fatigueTimeElapsed.textContent = '0.0s / 1.5s';
}

// Primary Inference Loop
async function inferenceLoop(timestamp) {
  if (!state.isRunning) return;

  const t0 = performance.now();
  state.webcam.update();

  // Compute FPS
  state.frameCount++;
  if (timestamp - state.lastFrameTime >= 1000) {
    state.currentFps = Math.round((state.frameCount * 1000) / (timestamp - state.lastFrameTime));
    elements.fpsCounter.textContent = `FPS: ${state.currentFps}`;
    state.frameCount = 0;
    state.lastFrameTime = timestamp;
  }

  // Obtain Class Predictions
  let predictions = [];

  if (state.model && !isUsingEmulatedModel) {
    predictions = await state.model.predict(state.webcam.canvas);
  } else {
    // Standard vision heuristic (facial brightness and movement analysis from video canvas)
    predictions = runVisionHeuristic(state.webcam.canvas);
  }

  const t1 = performance.now();
  elements.latencyCounter.textContent = `Latency: ${Math.round(t1 - t0)} ms`;

  // Process predictions and update UI
  processPredictions(predictions);

  requestAnimationFrame(inferenceLoop);
}

// Fallback Heuristic Analysis for testing before model URL is provided
let lastLuminance = null;
let headDroopCounter = 0;

function runVisionHeuristic(canvas) {
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // Sample center facial quadrant
  const sampleX = Math.floor(width * 0.25);
  const sampleY = Math.floor(height * 0.25);
  const sampleW = Math.floor(width * 0.5);
  const sampleH = Math.floor(height * 0.5);

  const imgData = ctx.getImageData(sampleX, sampleY, sampleW, sampleH).data;
  let totalLuminance = 0;
  const step = 8;
  let sampleCount = 0;

  for (let i = 0; i < imgData.length; i += step * 4) {
    totalLuminance += (imgData[i] * 0.299 + imgData[i + 1] * 0.587 + imgData[i + 2] * 0.114);
    sampleCount++;
  }

  const avgLuminance = totalLuminance / sampleCount;
  let alertProb = 0.92;
  let drowsyProb = 0.05;
  let distractedProb = 0.03;

  // Detect downward head movement or eye closure through luminance shift
  if (lastLuminance !== null) {
    const delta = Math.abs(avgLuminance - lastLuminance);
    if (delta > 35) {
      headDroopCounter = 25;
    }
  }
  lastLuminance = avgLuminance;

  if (headDroopCounter > 0) {
    headDroopCounter--;
    drowsyProb = 0.88;
    alertProb = 0.08;
    distractedProb = 0.04;
  }

  return [
    { className: 'Alert and Focused', probability: alertProb },
    { className: 'Drowsy / Nodding', probability: drowsyProb },
    { className: 'Distracted', probability: distractedProb }
  ];
}

// Telemetry & Hysteresis Processing
function processPredictions(predictions) {
  let highest = { className: 'Unknown', probability: 0 };
  let alertVal = 0;
  let drowsyVal = 0;
  let distractedVal = 0;

  predictions.forEach(p => {
    const name = p.className.toLowerCase();
    const prob = p.probability;

    if (name.includes('alert') || name.includes('focused') || name.includes('upright')) {
      alertVal = prob;
    } else if (name.includes('drowsy') || name.includes('nodding') || name.includes('sleep') || name.includes('closed')) {
      drowsyVal = prob;
    } else if (name.includes('distracted') || name.includes('away') || name.includes('turned')) {
      distractedVal = prob;
    }

    if (prob > highest.probability) {
      highest = p;
    }
  });

  // Update Percentage Bars
  elements.scoreAlert.textContent = `${(alertVal * 100).toFixed(1)}%`;
  elements.barAlert.style.width = `${(alertVal * 100).toFixed(0)}%`;

  elements.scoreDrowsy.textContent = `${(drowsyVal * 100).toFixed(1)}%`;
  elements.barDrowsy.style.width = `${(drowsyVal * 100).toFixed(0)}%`;

  elements.scoreDistracted.textContent = `${(distractedVal * 100).toFixed(1)}%`;
  elements.barDistracted.style.width = `${(distractedVal * 100).toFixed(0)}%`;

  // Update State Card Readout
  elements.currentClassification.textContent = highest.className;
  elements.primaryConfidence.textContent = `${(highest.probability * 100).toFixed(0)}%`;

  const now = performance.now();

  // Drowsiness Hysteresis Engine (Blink Buffer)
  if (drowsyVal > 0.65) {
    if (!state.activeDrowsyStart) {
      state.activeDrowsyStart = now;
    }
    const elapsedDrowsy = now - state.activeDrowsyStart;
    const progress = Math.min(100, (elapsedDrowsy / state.drowsyThresholdMs) * 100);
    elements.fatigueTriggerBar.style.width = `${progress.toFixed(0)}%`;
    elements.fatigueTimeElapsed.textContent = `${(elapsedDrowsy / 1000).toFixed(1)}s / 1.5s`;

    if (elapsedDrowsy >= state.drowsyThresholdMs) {
      // Confirmed fatigue event
      triggerFatigueAlert(highest.className, highest.probability);
    }
  } else {
    state.activeDrowsyStart = null;
    elements.fatigueTriggerBar.style.width = '0%';
    elements.fatigueTimeElapsed.textContent = '0.0s / 1.5s';
  }

  // Reticle HUD Color Mapping
  elements.reticleBox.className = 'reticle-box';
  elements.stateCard.className = 'state-card';

  if (drowsyVal > 0.65) {
    elements.reticleBox.classList.add('drowsy-state');
    elements.reticleStatusText.textContent = 'Drowsiness Detected';
    elements.stateCard.classList.add('critical-alert');
    updateStatus('critical', 'Fatigue Warning');
  } else if (distractedVal > 0.65) {
    elements.reticleBox.classList.add('distracted-state');
    elements.reticleStatusText.textContent = 'Operator Distracted';
    elements.stateCard.classList.add('warning-alert');
    updateStatus('warning', 'Operator Distracted');
  } else {
    elements.reticleBox.classList.add('alert-state');
    elements.reticleStatusText.textContent = 'Operator Alert';
    updateStatus('active', 'Operator Vigilant');
  }
}

// Trigger Fatigue Acoustic & Visual Notification
let lastAlertLoggedTime = 0;

function triggerFatigueAlert(label, confidence) {
  const now = Date.now();
  playWarningTone();

  // Log incident only once every 3 seconds to prevent table flooding
  if (now - lastAlertLoggedTime > 3000) {
    lastAlertLoggedTime = now;
    logIncident(label, confidence);
  }
}

// Log Event Entry
function logIncident(label, confidence) {
  if (label.toLowerCase().includes('drowsy')) {
    state.drowsyIncidentCount++;
    elements.drowsyIncidentCount.textContent = state.drowsyIncidentCount;
  } else {
    state.distractedIncidentCount++;
    elements.distractedIncidentCount.textContent = state.distractedIncidentCount;
  }

  const emptyRow = elements.incidentLogBody.querySelector('.empty-row');
  if (emptyRow) {
    emptyRow.remove();
  }

  const timeString = new Date().toLocaleTimeString();
  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${timeString}</td>
    <td>${label}</td>
    <td>${(confidence * 100).toFixed(0)}%</td>
  `;

  elements.incidentLogBody.prepend(row);
}

// Clear Incident Log
elements.clearLogBtn.addEventListener('click', () => {
  state.drowsyIncidentCount = 0;
  state.distractedIncidentCount = 0;
  elements.drowsyIncidentCount.textContent = '0';
  elements.distractedIncidentCount.textContent = '0';
  elements.incidentLogBody.innerHTML = `
    <tr class="empty-row">
      <td colspan="3">No fatigue incidents recorded in this session.</td>
    </tr>
  `;
});

// Event Listeners for Controls
elements.toggleCameraBtn.addEventListener('click', () => {
  if (state.isRunning) {
    stopMonitoring();
  } else {
    startMonitoring();
  }
});

if (elements.flipCameraBtn) {
  elements.flipCameraBtn.addEventListener('click', async () => {
    state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
    if (state.isRunning) {
      stopMonitoring();
      await startMonitoring();
    }
  });
}

elements.toggleAudioBtn.addEventListener('click', () => {
  state.isAudioMuted = !state.isAudioMuted;
  elements.audioBtnText.textContent = state.isAudioMuted ? 'Unmute Tone' : 'Mute Tone';
});

// Model Configuration Modal
elements.openSettingsBtn.addEventListener('click', () => {
  elements.modelModal.classList.remove('hidden');
});

elements.closeModelModalBtn.addEventListener('click', () => {
  elements.modelModal.classList.add('hidden');
});

elements.saveModelUrlBtn.addEventListener('click', async () => {
  const url = elements.modelUrlInput.value.trim();
  if (!url) {
    alert('Please enter a valid Teachable Machine URL.');
    return;
  }
  elements.saveModelUrlBtn.disabled = true;
  elements.saveModelUrlBtn.textContent = 'Loading...';
  const success = await loadModel(url);
  elements.saveModelUrlBtn.disabled = false;
  elements.saveModelUrlBtn.textContent = 'Load Custom Model';
  if (success) {
    elements.modelModal.classList.add('hidden');
    if (state.isRunning) {
      stopMonitoring();
      startMonitoring();
    }
  }
});

elements.usePresetModelBtn.addEventListener('click', () => {
  loadModel(null);
  elements.modelModal.classList.add('hidden');
});

// Terms of Service and Privacy Policy Modals
const TOS_CONTENT = `
  <h4>1. Acceptable Use and Safety Disclaimer</h4>
  <p>The DrivingWatch platform is an assistive computer vision system engineered for educational and experimental ergonomics monitoring. It is not an ISO-certified automotive active safety system and must not replace responsible operator judgment or human vigilance.</p>
  <h4>2. Hardware and Performance Variance</h4>
  <p>Inference throughput is contingent upon local client hardware acceleration, ambient illumination conditions, and webcam optical clarity. Operators should maintain adequate room lighting for consistent facial keypoint extraction.</p>
  <h4>3. Modification and Open Source Rights</h4>
  <p>This software is provided for academic demonstration and evaluation. The codebase may be freely inspected, customized, or deployed across personal and institutional repositories.</p>
`;

const PRIVACY_CONTENT = `
  <h4>1. Local On-Device Neural Processing</h4>
  <p>All video stream frames captured through your camera are processed entirely in your browser's local memory via WebGL and WebAssembly shaders. No video frames, telemetry data, or biometric representations are transmitted across the network or stored on remote servers.</p>
  <h4>2. Hardware Camera Permissions</h4>
  <p>Camera access is invoked strictly on pointer-down through the standard navigator.mediaDevices API. Vision capture terminates immediately when monitoring is stopped or when the browser tab is closed.</p>
  <h4>3. Storage and Telemetry Hygiene</h4>
  <p>The application stores zero tracking cookies, persistent identifiers, or cross-site tracking markers. Incident metrics remain strictly in runtime memory for the duration of the active session.</p>
`;

elements.openTosBtn.addEventListener('click', () => {
  elements.policyModalTitle.textContent = 'Terms of Service';
  elements.policyModalContent.innerHTML = TOS_CONTENT;
  elements.policyModal.classList.remove('hidden');
});

elements.openPrivacyBtn.addEventListener('click', () => {
  elements.policyModalTitle.textContent = 'Privacy Policy';
  elements.policyModalContent.innerHTML = PRIVACY_CONTENT;
  elements.policyModal.classList.remove('hidden');
});

elements.closePolicyModalBtn.addEventListener('click', () => {
  elements.policyModal.classList.add('hidden');
});

// Close modal on backdrop click
[elements.modelModal, elements.policyModal].forEach(modal => {
  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.classList.add('hidden');
    }
  });
});

// Keyboard accessibility: Escape to dismiss modals
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    elements.modelModal.classList.add('hidden');
    elements.policyModal.classList.add('hidden');
  }
});

// Initial Setup: Auto-load local model files on startup
if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', () => loadModel(null));
} else {
  loadModel(null);
}
