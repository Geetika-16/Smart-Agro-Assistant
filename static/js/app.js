/* ═══════════════════════════════════════════════════════
   SMART AGRO ASSISTANT — Shared JavaScript
   Features: Voice (TTS/STT), Language, Leaves, Toast
   ═══════════════════════════════════════════════════════ */

// ─── Language Map ─────────────────────────────────────
let LANGUAGES = {};
if (typeof ALL_LANGUAGES !== "undefined") {

    ALL_LANGUAGES.forEach(lang => {

        LANGUAGES[lang.code] = {
            code: lang.code,
            native: lang.native,
            name: lang.name,
            speech_code: lang.speech_code || lang.code, // BCP-47 for mic
            tts_code:    lang.tts_code    || lang.code  // BCP-47 for speaker
        };

    });

}

// Full lang map fetched from /api/languages (has Speech_Code + TTS_Code)
// Keyed by Translate_code e.g. { "ta": { code:"ta-IN", native:"தமிழ்" }, ... }
let _fullLangMap = null;
 
async function getFullLangMap() {
    if (_fullLangMap) return _fullLangMap;
    try {
        const res = await fetch("/api/languages");
        _fullLangMap = await res.json();
    } catch(e) {
        console.warn("Could not load language map:", e);
        _fullLangMap = {};
    }
    return _fullLangMap;
}
 
// Get BCP-47 speech locale for current user language
// e.g. "ta" → "ta-IN"
async function getSpeechLocale() {
    const map  = await getFullLangMap();
    const code = AgroState.lang;
    // /api/languages returns { "ta": { code: "ta-IN", native: "தமிழ்" } }
    return (map[code] && map[code].code) || code;
}

// ─── State ────────────────────────────────────────────
const AgroState = {
  lang:          localStorage.getItem('agro_lang') || 'en',
  isListening:   false,
  isSpeaking:    false,
  recognition:   null,
  synthesis:     window.speechSynthesis || null,

  get langInfo() {

    return LANGUAGES[this.lang] || {
        code: "en",
        native: "English",
        name: "English"
    };

},
  set(key, val) { this[key] = val; if(key === 'lang') localStorage.setItem('agro_lang', val);
    setGTranslateCookie(val); // ← keep cookie in sync whenever lang changes
  }
};

// ─── Init on DOM Ready ────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  //loadLanguages();
  initLeaves();
  initVoiceBtn();
  initLangDisplay();
  initToast();
  initTabs();
  setTimeout(() => {
    initDragDrop();
}, 100);
});

// ════════════════════════════════════════════════════════
//  GOOGLE TRANSLATE INTEGRATION
// ════════════════════════════════════════════════════════
 
/**
 * Sets the 'googtrans' cookie that the Google Translate widget reads
 * automatically when the page loads.  This is what makes translation
 * PERSIST across every page navigation.
 *
 * Format Google Translate expects:  /en/{targetLangCode}
 * e.g.  Tamil → /en/ta
 *       Hindi → /en/hi
 */
function setGTranslateCookie(langCode) {
    const COOKIE = 'googtrans';
    const FAR    = 'expires=Fri, 31 Dec 2049 23:59:59 GMT';
 
    if (!langCode || langCode === 'en') {
        // Clear cookie → revert to English
        document.cookie = `${COOKIE}=; path=/; ${FAR}`;
        try {
            const h = window.location.hostname;
            if (h) document.cookie = `${COOKIE}=; path=/; domain=.${h}; ${FAR}`;
        } catch (_) {}
        return;
    }
 
    const val = `/en/${langCode}`;
 
    // 1. Plain path cookie (works on localhost)
    document.cookie = `${COOKIE}=${val}; path=/; ${FAR}`;
 
    // 2. Domain-level cookie (required for Google Translate on real domains)
    try {
        const h = window.location.hostname;
        if (h && h !== 'localhost') {
            document.cookie = `${COOKIE}=${val}; path=/; domain=.${h}; ${FAR}`;
        }
    } catch (_) {}
}
 
/**
 * Triggers Google Translate to switch language on the CURRENT page.
 * Strategy:
 *   a) Set the cookie  → future page loads auto-translate
 *   b) Find the hidden .goog-te-combo dropdown and update it → current page translates now
 * Retries up to 20 times (every 500 ms = 10 s total) because the widget loads asynchronously.
 */
function triggerTranslation(langCode) {
    if (!langCode || langCode === 'en') return;
 
    setGTranslateCookie(langCode); // ← step (a)
 
    let tries = 0;
    function tryCombo() {
        const combo = document.querySelector('.goog-te-combo');
        if (combo) {
            // Widget is ready — set the language
            if (combo.value !== langCode) {
                combo.value = langCode;
                combo.dispatchEvent(new Event('change'));
            }
            return;
        }
        // Widget not ready yet — retry
        if (++tries < 20) setTimeout(tryCombo, 500);
    }
    tryCombo(); // ← step (b)
}

// ─── Floating Leaves ──────────────────────────────────
function initLeaves() {
  const container = document.querySelector('.leaves-container');
  if (!container) return;
  const emojis = ['🌿','🍃','🌱','🌾','🍀','🌻','🌼'];
  for (let i = 0; i < 12; i++) {
    const leaf = document.createElement('div');
    leaf.className = 'leaf';
    leaf.textContent = emojis[Math.floor(Math.random() * emojis.length)];
    leaf.style.cssText = `
      left: ${Math.random()*100}%;
      animation-duration: ${8 + Math.random()*12}s;
      animation-delay: ${-Math.random()*15}s;
      font-size: ${.8 + Math.random()*.9}rem;
      opacity: ${.4 + Math.random()*.4};
    `;
    container.appendChild(leaf);
  }
}

// ─── Language Display ─────────────────────────────────
function initLangDisplay() {
  const el = document.getElementById('selected-lang-display');
  if (el) {
    const info = AgroState.langInfo;
    el.textContent = info.native;
  }
  // Auto-trigger translation using the saved language
    const lang = AgroState.lang;
    if (lang && lang !== 'en') {
        triggerTranslation(lang);
    }
}

// ─── Toast ────────────────────────────────────────────
let toastEl;
function initToast() {
  toastEl = document.getElementById('toast');
  if (!toastEl) {
    toastEl = document.createElement('div');
    toastEl.id = 'toast';
    toastEl.className = 'toast';
    document.body.appendChild(toastEl);
  }
}
function showToast(msg, duration = 2800) {
  if (!toastEl) initToast();
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  setTimeout(() => toastEl.classList.remove('show'), duration);
}

// ─── Voice Button ─────────────────────────────────────
function initVoiceBtn() {
  const btn = document.querySelector('.voice-btn');
  if (!btn) return;
  btn.addEventListener('click', toggleVoiceAssistant);
  btn.title = 'Voice Assistant';
}

function toggleVoiceAssistant() {
  if (AgroState.isSpeaking) {
    stopSpeaking();
    return;
  }
  if (AgroState.isListening) {
    stopListening();
  } else {
    // Read the main heading first, then start listening
    const heading = document.querySelector('h1, .section-title, .page-hero h1');
    if (heading) {
      speak(`${heading.textContent}. Say a command or tap a button.`);
    } else {
      startListening();
    }
  }
}

// ─── Text-to-Speech ───────────────────────────────────
// ─── Text-to-Speech ───────────────────────────────────
// FIX: picks the correct BCP-47 voice from dataset via /api/languages
async function speak(text, forceLang) {
  if (!window.speechSynthesis) {
    showToast('🔇 Speech not supported in this browser');
    return;
  }
  window.speechSynthesis.cancel();
  if (!text || !text.trim()) return;
 
  // Use forceLang if provided, otherwise look up from dataset
  const locale = forceLang || await getSpeechLocale();
 
  const utter = new SpeechSynthesisUtterance(text.trim());
  utter.lang   = locale;   // ← NOW uses "ta-IN" not just "ta"
  utter.rate   = 0.95;
  utter.pitch  = 1.0;
  utter.volume = 1.0;
 
  function doSpeak() {
    const voices = window.speechSynthesis.getVoices();
    if (voices.length > 0) {
      // Try exact match first, then language-prefix match
      const exact  = voices.find(v => v.lang === locale);
      const prefix = voices.find(v => v.lang.startsWith(locale.split('-')[0]));
      if (exact)       utter.voice = exact;
      else if (prefix) utter.voice = prefix;
    }
    utter.onstart = () => {
      AgroState.isSpeaking = true;
      updateVoiceBtn(true);
      showToast('🔊 Reading aloud...');
    };
    utter.onend = utter.onerror = () => {
      AgroState.isSpeaking = false;
      updateVoiceBtn(false);
    };
    window.speechSynthesis.speak(utter);
  }
 
  // Voices may not be loaded yet on first call
  if (window.speechSynthesis.getVoices().length === 0) {
    window.speechSynthesis.onvoiceschanged = doSpeak;
  } else {
    doSpeak();
  }
}
 
function stopSpeaking() {
  window.speechSynthesis && window.speechSynthesis.cancel();
  AgroState.isSpeaking = false;
  updateVoiceBtn(false);
}
 
// Read aloud from element id — works even if element is hidden
function readAloud(elementId) {
  const el = document.getElementById(elementId);
  if (!el) return;
  readAloudText(el.textContent || el.innerText);
}
 
function readAloudText(text) {
  speak(text);
}

// ─── Speech Recognition ───────────────────────────────
// FIX: rec.lang now uses full BCP-47 from dataset e.g. "ta-IN"
async function startListening(callback) {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    showToast('🎤 Mic not supported in this browser. Use Chrome.');
    return;
  }
  if (AgroState.recognition) AgroState.recognition.stop();
 
  // Get full BCP-47 locale from dataset
  const locale = await getSpeechLocale();  // e.g. "ta-IN", "hi-IN"
 
  const rec = new SpeechRec();
  rec.lang             = locale;   // ← THIS was the bug — was just "ta" before
  rec.interimResults   = false;
  rec.maxAlternatives  = 1;
 
  AgroState.recognition = rec;
  AgroState.isListening = true;
  updateVoiceBtn(true, true);
  showToast('🎤 Listening in ' + locale + '… speak now');
 
  rec.onresult = (e) => {
    const transcript = e.results[0][0].transcript;
    showToast(`🗣️ "${transcript}"`);
    if (callback) callback(transcript);
    else handleVoiceCommand(transcript.toLowerCase());
    stopListening();
  };
  rec.onerror = (e) => {
    const msgs = {
      'no-speech':     'No speech detected. Try again.',
      'audio-capture': 'Microphone not found.',
      'not-allowed':   'Microphone access denied. Allow it in browser settings.',
      'network':       'Network error during voice recognition.'
    };
    showToast('⚠️ ' + (msgs[e.error] || 'Voice error: ' + e.error));
    AgroState.isListening = false;
    updateVoiceBtn(false);
  };
  rec.onend = () => {
    AgroState.isListening = false;
    updateVoiceBtn(false);
  };
  rec.start();
}
 
function stopListening() {
  if (AgroState.recognition) {
    AgroState.recognition.stop();
    AgroState.recognition = null;
  }
  AgroState.isListening = false;
  updateVoiceBtn(false);
}
 
function handleVoiceCommand(cmd) {
  if (cmd.includes('disease') || cmd.includes('leaf') || cmd.includes('plant')) {
    window.location.href = '/disease';
  } else if (cmd.includes('crop') || cmd.includes('recommend') || cmd.includes('grow')) {
    window.location.href = '/crop';
  } else if (cmd.includes('report') || cmd.includes('history')) {
    window.location.href = '/reports';
  } else if (cmd.includes('logout') || cmd.includes('sign out')) {
    window.location.href = '/logout';
  } else if (cmd.includes('home') || cmd.includes('dashboard')) {
    window.location.href = '/dashboard';
  } else {
    speak('I heard: ' + cmd + '. Please try: disease detection, crop recommendation, or reports.');
  }
}
 
function updateVoiceBtn(active, listening = false) {
  const btn = document.querySelector('.voice-btn');
  if (!btn) return;
  btn.classList.toggle('listening', active);
  btn.innerHTML = active ? (listening ? '🎤' : '🔊') : '🎙️';
}

// ─── Tab System ───────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.dataset.group || btn.closest('.tab-bar')?.dataset.group;
      const target = btn.dataset.tab;
      document.querySelectorAll(`.tab-btn[data-group="${group}"]`).forEach(b => b.classList.remove('active'));
      document.querySelectorAll(`.tab-content[data-group="${group}"]`).forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const content = document.querySelector(`.tab-content[data-tab="${target}"]`);
      if (content) content.classList.add('active');
    });
  });
}

// ─── Drag & Drop Upload ───────────────────────────────
function initDragDrop() {
  const zones = document.querySelectorAll('.upload-zone');
  zones.forEach(zone => {
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) handleFileUpload(file, zone);
    });
    const inp = zone.querySelector('input[type=file]');
    if (inp) {

  inp.onchange = function(e) {
    const file = e.target.files[0];
    if (file) {
      handleFileUpload(file, zone);
    }
  };

  zone.onclick = function(e) {

    if (
      e.target.tagName !== 'BUTTON' &&
      e.target.tagName !== 'INPUT'
    ) {
      inp.click();
    }

  };

}
  });
}
function handleFileUpload(file, zone) {
  if (!file.type.startsWith('image/')) {
    showToast('⚠️ Please upload an image file');
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    zone.innerHTML = `
    <input type="file" id="leaf-file" accept="image/*" />
    
    <img src="${e.target.result}"
       style="max-height:200px;border-radius:12px;margin:0 auto"
       alt="Uploaded leaf">
       
    <p style="margin-top:12px;color:var(--green-mid);font-weight:700">
     ✓ Image ready — click Analyze below
  </p>
`;
    zone.dataset.imageData = e.target.result;
    initDragDrop();
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) analyzeBtn.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

// ─── Camera ───────────────────────────────────────────
let cameraStream = null;

async function startCamera() {
  const video = document.getElementById('camera-feed');

  if (!video) return;

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: true
    });

    video.srcObject = cameraStream;

    // SHOW VIDEO
    video.style.display = 'block';

    // HIDE PLACEHOLDER
    document.getElementById('cam-placeholder').style.display = 'none';

    await video.play();

    document.getElementById('capture-btn').classList.remove('hidden');
    document.getElementById('stop-cam-btn').classList.remove('hidden');
    document.getElementById('start-cam-btn').classList.add('hidden');

    showToast('📷 Camera started');

  } catch (err) {
    console.error(err);
    showToast('❌ Camera access denied');
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }

  const video = document.getElementById('camera-feed');

  if (video) {
    video.srcObject = null;
    video.style.display = 'none';
  }

  document.getElementById('cam-placeholder').style.display = 'flex';

  document.getElementById('capture-btn').classList.add('hidden');
  document.getElementById('stop-cam-btn').classList.add('hidden');
  document.getElementById('start-cam-btn').classList.remove('hidden');
}
function capturePhoto() {
  const video = document.getElementById('camera-feed');
  if (!video) return;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const dataUrl = canvas.toDataURL('image/jpeg');
  stopCamera();
  const preview = document.getElementById('capture-preview');
  if (preview) {
    preview.src = dataUrl;
    preview.classList.remove('hidden');
  }
  const zone = document.querySelector('.upload-zone');
  if (zone) zone.dataset.imageData = dataUrl;
  const analyzeBtn = document.getElementById('analyze-btn');
  if (analyzeBtn) analyzeBtn.classList.remove('hidden');
  showToast('📸 Photo captured!');
}

// ─── API Helpers ──────────────────────────────────────
async function apiPost(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return res.json();
}

// ─── Animate numbers ──────────────────────────────────
function animateCount(el, target, duration = 1200) {
  const start = performance.now();
  const update = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    el.textContent = Math.floor(progress * target);
    if (progress < 1) requestAnimationFrame(update);
    else el.textContent = target;
  };
  requestAnimationFrame(update);
}

// ─── Export ───────────────────────────────────────────
window.AgroApp = {
  speak, stopSpeaking, startListening, stopListening,
  readAloud, readAloudText, showToast,
  startCamera, stopCamera, capturePhoto,
  handleFileUpload, animateCount, 
  setGTranslateCookie, triggerTranslation,   // ← new exports
  AgroState, LANGUAGES
};
