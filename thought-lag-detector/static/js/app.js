// Clean implementation (reconstructed)
let recorder;
// Session lifecycle flags
let sessionActive = false; // true between Start and Stop (or natural completion)
let nextPromptTimer = null; // timeout id for scheduling next prompt
let sessionId = null;
let clientId = localStorage.getItem("tld_client_id") || crypto.randomUUID();
localStorage.setItem("tld_client_id", clientId);
let idx = 0,
  audioCtx,
  tPromptEndMs = 0,
  tOnsetMs = 0,
  reactionMs = 0,
  autoSubmitTimer = null,
  submitting = false;
const RESPONSE_WINDOW_MS = 2200; // shorter fallback window for faster cycles
let clientSeq = 0;
const sendTimes = new Map(); // client_seq -> timestamp ms
const rolling = { stress: [], focus: [], reaction: [], max: 20 };

// Elements
const promptBox = document.getElementById("promptBox");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const baselineToggle = document.getElementById("baselineToggle");
const fastModeToggle = document.getElementById("fastModeToggle");
const reactionSpan = document.getElementById("reactionVal");
const stressSpan = document.getElementById("stressVal");
const focusSpan = document.getElementById("focusVal");
const mlStressSpan = document.getElementById("mlStressVal");
const sylRateSpan = document.getElementById("sylRateVal");
const pauseRatioSpan = document.getElementById("pauseRatioVal");
const pitchSpan = document.getElementById("avgPitchVal");
const procSpan = document.getElementById("procMsVal");
// Add optional new average spans if present
const avgStressSpan = document.getElementById("avgStressVal");
const avgFocusSpan = document.getElementById("avgFocusVal");
const avgReactionSpan = document.getElementById("avgReactionVal");
const latencySpan = document.getElementById("latencyVal");
const sessionHistory = document.getElementById("sessionHistory");
const latestTranscript = document.getElementById("latestTranscript");
const sessionIdDisplay = document.getElementById("sessionIdDisplay");
const volPct = document.getElementById("volPct");
const miniWave = document.getElementById("miniWave");
const emotionSpan = document.getElementById("emotionVal");
// Analytics elements
const divergenceVal = document.getElementById("divergenceVal");
const mlSampleCounts = document.getElementById("mlSampleCounts");
const mlTrainedFlag = document.getElementById("mlTrainedFlag");

// Charts
const reactionChart = new Chart(
  document.getElementById("reactionChart").getContext("2d"),
  {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Reaction (ms)",
          data: [],
          borderColor: "#3b82f6",
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      animation: false,
      scales: {
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
        x: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" } },
      },
      plugins: { legend: { labels: { color: "#e2e8f0" } } },
    },
  }
);
const stressTrendChart = new Chart(
  document.getElementById("stressTrendChart").getContext("2d"),
  {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Stress", data: [], borderColor: "#f87171", tension: 0.3 },
      ],
    },
    options: {
      responsive: true,
      animation: false,
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { color: "#94a3b8" },
          grid: { color: "#1e293b" },
        },
        x: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" } },
      },
      plugins: { legend: { labels: { color: "#e2e8f0" } } },
    },
  }
);
const focusTrendChart = new Chart(
  document.getElementById("focusTrendChart").getContext("2d"),
  {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Focus", data: [], borderColor: "#34d399", tension: 0.3 },
      ],
    },
    options: {
      responsive: true,
      animation: false,
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { color: "#94a3b8" },
          grid: { color: "#1e293b" },
        },
        x: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" } },
      },
      plugins: { legend: { labels: { color: "#e2e8f0" } } },
    },
  }
);

// Waveform
const miniWaveCtx = miniWave.getContext("2d");
function drawLevel(level) {
  miniWaveCtx.fillStyle = "#000";
  miniWaveCtx.fillRect(0, 0, miniWave.width, miniWave.height);
  const h = miniWave.height,
    y = h / 2;
  miniWaveCtx.strokeStyle = "#00ff3b";
  miniWaveCtx.lineWidth = 2;
  miniWaveCtx.beginPath();
  miniWaveCtx.moveTo(0, y);
  miniWaveCtx.lineTo(miniWave.width, y);
  miniWaveCtx.stroke();
  miniWaveCtx.fillStyle = "#16a34a";
  miniWaveCtx.fillRect(0, y - (h * level) / 2, miniWave.width, h * level);
}

// Socket helpers
let socket = null;
const prSeen = new Set();
const MAX_POINTS = 200;
function trimSeries(chart) {
  const ds = chart.data.datasets[0];
  if (ds.data.length > MAX_POINTS) {
    const over = ds.data.length - MAX_POINTS;
    ds.data.splice(0, over);
    chart.data.labels.splice(0, over);
  }
}
function setStatus(text, cls) {
  ["socketStatus", "socketOverlayStatus"].forEach((id) => {
    const e = document.getElementById(id);
    if (!e) return;
    e.textContent = text;
    e.classList.remove("online", "offline");
    if (cls) e.classList.add(cls);
  });
}
function ensureSocket() {
  if (socket || typeof io === "undefined") return;
  socket = io({
    autoConnect: false,
    reconnectionAttempts: 8,
    reconnectionDelay: 1200,
  });
  socket.on("connect", () => {
    setStatus("Connected", "online");
    socket.emit("register", { client_id: clientId });
    // expose lightweight emit helper for recorder streaming
    window._tldSocketEmit = (ev, payload) => {
      try {
        socket.emit(ev, payload);
      } catch (e) {}
    };
  });
  socket.on("disconnect", () => setStatus("Disconnected", "offline"));
  socket.on("analysis_result", (p) => applyResult(p, true));
  socket.on("provisional_score", (p) => {
    if (!p || !sessionActive) return;
    if (p.scores && p.scores.stress != null) {
      // Show faint provisional overlay: use last label index
      stressSpan.textContent = p.scores.stress.toFixed
        ? p.scores.stress.toFixed(1)
        : p.scores.stress;
      focusSpan.textContent = p.scores.focus.toFixed
        ? p.scores.focus.toFixed(1)
        : p.scores.focus;
    }
  });
  socket.on("live_update", (p) => {
    if (!p || !sessionActive) return;
    if (
      p.partial_transcript &&
      latestTranscript.textContent.startsWith("(waiting")
    ) {
      latestTranscript.textContent = p.partial_transcript;
    } else if (p.partial_transcript) {
      // update live
      latestTranscript.textContent = p.partial_transcript;
    }
    // derive quick emotion using simple heuristic on current text length & rms changes
    if (emotionSpan && typeof p.rms === "number") {
      const lvl = p.rms;
      if (lvl > 0.09) emotionSpan.textContent = "energized";
      else if (lvl > 0.04) emotionSpan.textContent = "engaged";
      else emotionSpan.textContent = "calm";
    }
  });
}

function applyResult(payload, fromSocket = false) {
  if (!payload || !payload.pr_id) return;
  if (prSeen.has(payload.pr_id)) return;
  prSeen.add(payload.pr_id);
  const {
    scores = {},
    features = {},
    transcript,
    processing_ms,
    prompt_idx,
    client_seq,
    fast_mode,
  } = payload;
  if (scores.stress != null)
    stressSpan.textContent = scores.stress.toFixed
      ? scores.stress.toFixed(1)
      : scores.stress;
  if (scores.focus != null)
    focusSpan.textContent = scores.focus.toFixed
      ? scores.focus.toFixed(1)
      : scores.focus;
  if (scores.ml_stress_prob != null && mlStressSpan) {
    mlStressSpan.textContent = scores.ml_stress_prob.toFixed
      ? scores.ml_stress_prob.toFixed(1)
      : scores.ml_stress_prob;
  }
  const rx = Math.round(
    features.reaction_ms_server ||
      features.reaction_ms_client ||
      reactionMs ||
      0
  );
  reactionSpan.textContent = rx;
  // Track rolling averages
  if (!isNaN(scores.stress)) {
    rolling.stress.push(scores.stress);
    if (rolling.stress.length > rolling.max) rolling.stress.shift();
  }
  if (!isNaN(scores.focus)) {
    rolling.focus.push(scores.focus);
    if (rolling.focus.length > rolling.max) rolling.focus.shift();
  }
  if (!isNaN(rx)) {
    rolling.reaction.push(rx);
    if (rolling.reaction.length > rolling.max) rolling.reaction.shift();
  }
  if (avgStressSpan)
    avgStressSpan.textContent = (
      rolling.stress.reduce((a, b) => a + b, 0) /
      Math.max(rolling.stress.length, 1)
    ).toFixed(1);
  if (avgFocusSpan)
    avgFocusSpan.textContent = (
      rolling.focus.reduce((a, b) => a + b, 0) /
      Math.max(rolling.focus.length, 1)
    ).toFixed(1);
  if (avgReactionSpan)
    avgReactionSpan.textContent = (
      rolling.reaction.reduce((a, b) => a + b, 0) /
      Math.max(rolling.reaction.length, 1)
    ).toFixed(0);
  if (latencySpan && client_seq != null && sendTimes.has(client_seq)) {
    const rtt = Date.now() - sendTimes.get(client_seq);
    latencySpan.textContent = rtt;
    sendTimes.delete(client_seq);
  }
  if (features.speaking_rate_wps != null)
    sylRateSpan.textContent = (features.speaking_rate_wps || 0).toFixed(1);
  if (features.pause_ratio != null)
    pauseRatioSpan.textContent = ((features.pause_ratio || 0) * 100).toFixed(0);
  if (features.pitch_mean_hz != null)
    pitchSpan.textContent = (features.pitch_mean_hz || 0).toFixed(0);
  if (processing_ms != null) procSpan.textContent = processing_ms;
  if (transcript != null)
    latestTranscript.textContent = transcript || "(empty)";
  // Driver list update
  const driverUl = document.getElementById("driverList");
  if (driverUl && scores.drivers) {
    driverUl.innerHTML = "";
    scores.drivers.forEach(([k, v]) => {
      const li = document.createElement("li");
      li.textContent = `${k} ${v.toFixed ? v.toFixed(2) : v}`;
      driverUl.appendChild(li);
    });
  }
  // Baseline bands: once baseline_means available add shaded region mean±std
  if (scores.baseline_means && scores.baseline_stds) {
    ensureBaselineBands(scores);
  }
  const chartIdx =
    typeof prompt_idx === "number"
      ? prompt_idx
      : reactionChart.data.labels.length;
  reactionChart.data.labels.push(chartIdx);
  reactionChart.data.datasets[0].data.push(rx);
  stressTrendChart.data.labels.push(chartIdx);
  stressTrendChart.data.datasets[0].data.push(scores.stress || 0);
  focusTrendChart.data.labels.push(chartIdx);
  focusTrendChart.data.datasets[0].data.push(scores.focus || 0);
  [reactionChart, stressTrendChart, focusTrendChart].forEach((c) => {
    c.update("none");
    trimSeries(c);
  });
  const li = document.createElement("li");
  if (fast_mode) li.classList.add("fast-mode");
  li.innerHTML = `<span>${new Date().toLocaleTimeString()} - ${
    PROMPTS[chartIdx] || "#"
  } </span><span style="color:#f87171;">Stress ${
    (scores.stress || 0).toFixed ? scores.stress.toFixed(0) : scores.stress
  }</span><span style="color:#34d399;">Focus ${
    (scores.focus || 0).toFixed ? scores.focus.toFixed(0) : scores.focus
  }</span>`;
  sessionHistory.prepend(li);
  while (sessionHistory.children.length > MAX_POINTS)
    sessionHistory.removeChild(sessionHistory.lastChild);
  if (emotionSpan) {
    if ((scores.stress || 0) > 70 && (scores.focus || 0) < 40)
      emotionSpan.textContent = "anxious";
    else if ((scores.focus || 0) > 70 && (scores.stress || 0) < 40)
      emotionSpan.textContent = "engaged";
    else emotionSpan.textContent = "neutral";
  }
  // Analytics divergence: absolute difference between heuristic stress and ML probability
  if (divergenceVal && scores.ml_stress_prob != null && scores.stress != null) {
    const div = Math.abs(scores.stress - scores.ml_stress_prob);
    divergenceVal.textContent = div.toFixed(1);
  }
  if (mlSampleCounts && scores.ml_pos != null && scores.ml_neg != null) {
    mlSampleCounts.textContent = `${scores.ml_pos} / ${scores.ml_neg}`;
  }
  if (mlTrainedFlag && scores.ml_trained != null) {
    mlTrainedFlag.textContent = scores.ml_trained ? "yes" : "no";
    mlTrainedFlag.style.color = scores.ml_trained ? "#34d399" : "#f87171";
  }
  // Highlight metric cards briefly for fast mode
  if (fast_mode) {
    const cards = document.querySelectorAll(".metric-card");
    cards.forEach((c) => c.classList.add("fast-mode"));
    setTimeout(
      () => cards.forEach((c) => c.classList.remove("fast-mode")),
      1200
    );
  }
  if (fromSocket) {
    try {
      socket.emit("ack", { pr_id: payload.pr_id, client_id: clientId });
    } catch (e) {}
  }
  if (!fromSocket && (!socket || !socket.connected))
    setStatus("HTTP fallback", "offline");
}

function ensureBaselineBands(scores) {
  try {
    const sMeans = scores.baseline_means;
    const sStds = scores.baseline_stds;
    if (!sMeans || !sStds) return;
    // Only add once
    if (!stressTrendChart.data.datasets.some((d) => d._baselineBand)) {
      const mean = sMeans.stress || sMeans["stress"]; // not stored directly (scores has stress scalar); use running average impossible; skip if missing
      // We don't have baseline mean for stress in means dict (feature-only), so approximate using rolling avg
      const avgStress = rolling.stress.length
        ? rolling.stress.reduce((a, b) => a + b, 0) / rolling.stress.length
        : null;
      if (avgStress != null) {
        const bandHi = new Array(stressTrendChart.data.labels.length).fill(
          avgStress + 5
        );
        const bandLo = new Array(stressTrendChart.data.labels.length).fill(
          avgStress - 5
        );
        stressTrendChart.data.datasets.push({
          label: "Baseline Band High",
          data: bandHi,
          borderWidth: 0,
          pointRadius: 0,
          backgroundColor: "rgba(248,113,113,0.10)",
          fill: "+1",
          _baselineBand: true,
        });
        stressTrendChart.data.datasets.push({
          label: "Baseline Band Low",
          data: bandLo,
          borderWidth: 0,
          pointRadius: 0,
          backgroundColor: "rgba(248,113,113,0.05)",
          fill: false,
          _baselineBand: true,
        });
      }
    }
    if (!focusTrendChart.data.datasets.some((d) => d._baselineBand)) {
      const avgFocus = rolling.focus.length
        ? rolling.focus.reduce((a, b) => a + b, 0) / rolling.focus.length
        : null;
      if (avgFocus != null) {
        const bandHi = new Array(focusTrendChart.data.labels.length).fill(
          avgFocus + 5
        );
        const bandLo = new Array(focusTrendChart.data.labels.length).fill(
          avgFocus - 5
        );
        focusTrendChart.data.datasets.push({
          label: "Baseline Band High",
          data: bandHi,
          borderWidth: 0,
          pointRadius: 0,
          backgroundColor: "rgba(52,211,153,0.10)",
          fill: "+1",
          _baselineBand: true,
        });
        focusTrendChart.data.datasets.push({
          label: "Baseline Band Low",
          data: bandLo,
          borderWidth: 0,
          pointRadius: 0,
          backgroundColor: "rgba(52,211,153,0.05)",
          fill: false,
          _baselineBand: true,
        });
      }
    }
  } catch (e) {}
}

function speakPrompt(text, onEnd) {
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.0;
  u.onend = () => {
    // If session was stopped mid-utterance, abort further actions
    if (!sessionActive) return;
    const ctx =
      audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    audioCtx = ctx;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = 880;
    g.gain.setValueAtTime(0.0001, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.25, ctx.currentTime + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.08);
    osc.connect(g);
    g.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.1);
    setTimeout(onEnd, 95);
  };
  speechSynthesis.speak(u);
}

async function startSession() {
  try {
    const resp = await fetch("/api/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        client_id: clientId,
        baseline: baselineToggle?.checked,
      }),
    });
    const data = await resp.json();
    sessionId = data.session_id;
    sessionIdDisplay.textContent = sessionId;
  } catch (e) {
    console.error("startSession", e);
  }
}

function initRecorder() {
  if (recorder) return;
  recorder = new WavRecorder({
    onData: (buf) => {
      let rms = 0;
      for (let i = 0; i < buf.length; i++) rms += buf[i] * buf[i];
      rms = Math.sqrt(rms / buf.length);
      const level = Math.min(1, rms * 12);
      volPct.textContent = Math.round(level * 100) + "%";
      drawLevel(level);
    },
    onVADOnset: (tMs) => {
      tOnsetMs = tMs;
      if (tOnsetMs < tPromptEndMs - 50) return;
      reactionMs = Math.max(0, Math.round(tOnsetMs - tPromptEndMs));
      reactionSpan.textContent = reactionMs;
    },
    onSpeechEnd: () => {
      // If user stops speaking early, submit immediately (if not already submitting)
      if (!submitting) submitResponse();
    },
  });
  recorder.start().catch((e) => console.error("recorder start", e));
}

async function runPrompt() {
  if (!sessionActive) return; // guard if stopped before scheduling
  if (idx >= PROMPTS.length) {
    promptBox.textContent = "Done!";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    sessionActive = false;
    return;
  }
  promptBox.textContent = PROMPTS[idx];
  // expose index for streaming meta
  window._tldPromptIndex = idx;
  initRecorder();
  ensureSocket();
  speakPrompt(PROMPTS[idx], () => {
    if (!sessionActive) return; // stopped while speaking
    tPromptEndMs = audioCtx.currentTime * 1000.0;
    recorder.arm();
    autoSubmitTimer = setTimeout(() => {
      if (!sessionActive) return; // don't auto-submit after stop
      submitResponse();
    }, RESPONSE_WINDOW_MS);
  });
}

async function submitResponse() {
  if (submitting) return;
  submitting = true;
  if (!recorder) {
    submitting = false;
    return;
  }
  if (autoSubmitTimer) {
    clearTimeout(autoSubmitTimer);
    autoSubmitTimer = null;
  }
  const wavBlob = recorder.exportWav();
  const form = new FormData();
  form.append("client_id", clientId);
  form.append("session_id", sessionId);
  form.append("prompt_text", PROMPTS[idx]);
  form.append("prompt_idx", idx);
  form.append("client_seq", clientSeq);
  if (fastModeToggle && fastModeToggle.checked) form.append("fast", "1");
  form.append("t_prompt_end_ms", tPromptEndMs);
  form.append("t_onset_ms", tOnsetMs);
  form.append("reaction_ms_client", reactionMs);
  form.append("sample_rate", recorder.sampleRate);
  form.append("audio", wavBlob, `resp_${idx}.wav`);
  let analyzeData = null;
  try {
    const resp = await fetch("/api/analyze", { method: "POST", body: form });
    sendTimes.set(clientSeq, Date.now());
    clientSeq++;
    if (resp.ok) {
      analyzeData = await resp.json();
      applyResult(analyzeData, false);
    } else console.error("analyze http", resp.status);
  } catch (e) {
    console.error("analyze", e);
  } finally {
    submitting = false;
  }
  idx++;
  if (sessionActive && idx < PROMPTS.length) {
    nextPromptTimer = setTimeout(() => {
      if (!sessionActive) return;
      runPrompt();
    }, 350);
  } else if (!sessionActive) {
    promptBox.textContent = "Session Stopped";
    startBtn.disabled = false;
    stopBtn.disabled = true;
  } else {
    promptBox.textContent = "Session Complete";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    sessionActive = false;
  }
}

function stopRecording() {
  if (recorder) {
    try {
      recorder.stop();
    } catch (e) {}
  }
  if (autoSubmitTimer) {
    clearTimeout(autoSubmitTimer);
    autoSubmitTimer = null;
  }
  if (nextPromptTimer) {
    clearTimeout(nextPromptTimer);
    nextPromptTimer = null;
  }
  if (socket) {
    try {
      socket.disconnect();
    } catch (e) {}
  }
  setStatus("Disconnected", "offline");
  startBtn.disabled = false;
  stopBtn.disabled = true;
  try {
    speechSynthesis.cancel();
  } catch (e) {}
}

startBtn.onclick = async () => {
  if (startBtn.disabled) return;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  sessionActive = true;
  await startSession();
  ensureSocket();
  try {
    socket.connect();
  } catch (e) {}
  idx = 0;
  prSeen.clear();
  reactionChart.data.labels = [];
  reactionChart.data.datasets[0].data = [];
  stressTrendChart.data.labels = [];
  stressTrendChart.data.datasets[0].data = [];
  focusTrendChart.data.labels = [];
  focusTrendChart.data.datasets[0].data = [];
  sessionHistory.innerHTML = "";
  runPrompt();
};
stopBtn.onclick = () => {
  if (stopBtn.disabled) return;
  stopBtn.disabled = true;
  // Mark session inactive first so no new prompts schedule.
  sessionActive = false;
  // Optionally finalize current response if already submitting; otherwise we skip to preserve immediate stop.
  if (recorder && !submitting) {
    // We could choose to submit current audio; for strict stop we skip submission.
    // submitResponse(); // <- intentionally disabled for hard stop
  }
  stopRecording();
  promptBox.textContent = "Session Stopped";
};
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !stopBtn.disabled) submitResponse();
});

// Log toggle
const logBtn = document.getElementById("socketLogToggle");
const logDiv = document.getElementById("socketLog");
if (logBtn)
  logBtn.onclick = () => {
    if (!logDiv) return;
    logDiv.hidden = !logDiv.hidden;
  };
