const sourceSelect = document.getElementById("sourceSelect");
const filePicker = document.querySelector(".file-picker");
const fileInput = document.getElementById("fileInput");
const fileLabel = document.getElementById("fileLabel");
const promptInput = document.getElementById("promptInput");
const startBtn = document.getElementById("startBtn");
const videoPlayer = document.getElementById("videoPlayer");
const audioPlayer = document.getElementById("audioPlayer");
const mediaClock = document.getElementById("mediaClock");
const playState = document.getElementById("playState");
const planView = document.getElementById("planView");
const messagesView = document.getElementById("messagesView");
const logsView = document.getElementById("logsView");
const messageCount = document.getElementById("messageCount");
const logCount = document.getElementById("logCount");
const sessionState = document.getElementById("sessionState");
const livePromptInput = document.getElementById("livePromptInput");
const sendPromptBtn = document.getElementById("sendPromptBtn");
const ttsToggle = document.getElementById("ttsToggle");
const previewPanel = document.getElementById("previewPanel");
const previewTime = document.getElementById("previewTime");
const previewImage = document.getElementById("previewImage");
const livePlayPauseBtn = document.getElementById("livePlayPauseBtn");
const audioSurface = document.getElementById("audioSurface");
const audioFileName = document.getElementById("audioFileName");
const audioPlayBtn = document.getElementById("audioPlayBtn");

const UPLOAD_SOURCE = "__upload__";
const LIVE_CAMERA_ID = "__live_camera__";
const LIVE_SCREEN_ID = "__live_screen__";
// The server reports the same constants for any live source; there is no file
// to measure, so the client can hold them rather than ask.
const LIVE_MEDIA_INFO = {
  audio_only: false,
  live: true,
  media_version: "live",
  file_size: null,
  mtime_ns: null,
  video_duration_ms: null,
  video_frame_count: null,
  video_fps: null
};
const LIVE_FRAME_INTERVAL_MS = 250;
const LIVE_FRAME_MAX_WIDTH = 640;
const LIVE_AUDIO_SAMPLE_RATE = 16000;
const LIVE_AUDIO_CHUNK_MS = 500;

let ws = null;
let sessionId = null;
let activePlayer = videoPlayer;
let messages = [];
let logs = [];
let currentGeneratedAudio = null;
let livePreviewStream = null;
let liveFrameTimer = null;
let liveFrameCanvas = null;
let liveFrameSequence = 0;
let liveAudioContext = null;
let liveAudioSourceNode = null;
let liveAudioProcessor = null;
let liveAudioChunks = [];
let liveAudioBufferedSamples = 0;
let liveAudioSequence = 0;
let activeLive = false;
let liveRunningSinceMs = null;
let liveElapsedBeforePauseMs = 0;
let backendStarted = false;
let backendStarting = false;
let planCompleted = false;
let playRequested = false;
let protectedMediaTime = 0;
let restoringProtectedTime = false;
let processingComplete = false;
let mediaInfo = null;
let planningInProgress = false;
let activeAudioOnly = false;

function formatMs(ms) {
  if (ms === null || ms === undefined || Number.isNaN(Number(ms))) return "PRE";
  const total = Math.max(0, Math.floor(Number(ms)));
  const minutes = Math.floor(total / 60000);
  const seconds = Math.floor((total % 60000) / 1000);
  const millis = total % 1000;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function currentMediaMs() {
  if (activeLive) {
    const runningMs = liveRunningSinceMs === null ? 0 : performance.now() - liveRunningSinceMs;
    return Math.max(0, Math.round(liveElapsedBeforePauseMs + runningMs));
  }
  const rawMs = Math.round((activePlayer?.currentTime || 0) * 1000);
  return clampDisplayedMediaMs(rawMs);
}

function updateClock() {
  mediaClock.textContent = formatMs(currentMediaMs());
  const state = processingComplete ? "complete" : activePlayer.paused ? "paused" : "playing";
  playState.textContent = sessionId ? state : "idle";
  const playLabel = planningInProgress ? "Planning..." : activePlayer.paused ? "Play" : "Pause";
  livePlayPauseBtn.textContent = playLabel;
  livePlayPauseBtn.disabled = planningInProgress || (activePlayer.ended && !processingComplete);
  audioPlayBtn.textContent = playLabel;
  audioPlayBtn.disabled = planningInProgress || (sessionId && !planCompleted) || (activePlayer.ended && !processingComplete);
  requestAnimationFrame(updateClock);
}

// The chosen source: a live id, or an uploaded file described by the server.
let selectedMedia = null;
// The last file uploaded this session, kept so switching away to a live source
// and back does not lose it.
let uploadedMedia = null;

function currentMediaId() {
  return selectedMedia?.id ?? null;
}

function currentIsLive() {
  return isLiveMedia(currentMediaId());
}

function currentIsAudioOnly() {
  return Boolean(selectedMedia?.audioOnly);
}

function currentMediaLabel() {
  if (currentIsLive()) return liveMediaLabel(currentMediaId());
  return selectedMedia?.name ?? "Media";
}

function syncSourceControls() {
  const uploading = sourceSelect.value === UPLOAD_SOURCE;
  filePicker.classList.toggle("is-hidden", !uploading);
  startBtn.disabled = !currentMediaId();
}

async function onSourceChange() {
  if (sourceSelect.value === UPLOAD_SOURCE) {
    selectedMedia = uploadedMedia;
    syncSourceControls();
    if (selectedMedia) {
      setupMedia(selectedMedia.url, selectedMedia.audioOnly, selectedMedia.info);
      sessionState.textContent = "media loaded";
    } else {
      sessionState.textContent = "choose a file";
    }
    renderPlan(null);
    return;
  }
  selectedMedia = { id: sourceSelect.value, name: liveMediaLabel(sourceSelect.value), audioOnly: false };
  syncSourceControls();
  mediaInfo = LIVE_MEDIA_INFO;
  await setupLiveMedia(mediaInfo);
  sessionState.textContent = `${liveMediaLabel(sourceSelect.value).toLowerCase()} loaded`;
  renderPlan(null);
}

async function onFileChosen() {
  const file = fileInput.files?.[0];
  if (!file) return;
  fileLabel.textContent = `Uploading ${file.name}...`;
  startBtn.disabled = true;
  sessionState.textContent = "uploading";
  const body = new FormData();
  body.append("file", file);
  let response;
  try {
    response = await fetch("/api/media", { method: "POST", body });
  } catch (error) {
    fileLabel.textContent = "Choose a file";
    sessionState.textContent = "upload failed";
    addLog({ wall_ms: 0, stream_time_ms: null, message: `Upload failed: ${error.message || error}` });
    return;
  }
  if (!response.ok) {
    fileLabel.textContent = "Choose a file";
    sessionState.textContent = "upload failed";
    addLog({ wall_ms: 0, stream_time_ms: null, message: await response.text() });
    return;
  }
  const data = await response.json();
  uploadedMedia = {
    id: data.media_id,
    name: data.name,
    audioOnly: data.audio_only,
    url: data.media_url,
    info: data.media_info
  };
  selectedMedia = uploadedMedia;
  fileLabel.textContent = data.name;
  mediaInfo = data.media_info;
  setupMedia(data.media_url, data.audio_only, data.media_info);
  sessionState.textContent = "media loaded";
  syncSourceControls();
  syncStartButton();
  renderPlan(null);
}

function renderPlan(plan) {
  if (!plan || !plan.gates) {
    planView.innerHTML = `<div class="empty">No plan yet.</div>`;
    return;
  }
  const gates = plan.gates
    .map((gate, index) => {
      const params = Object.entries(displayGateParams(gate))
        .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
        .join(", ");
      return `
        <div class="gate">
          <div class="gate-title">
            <strong>${index + 1}. ${escapeHtml(gate.id)}</strong>
            <span class="meta">${escapeHtml(gate.type)}</span>
          </div>
          <div class="params">${escapeHtml(params || "no params")}</div>
        </div>`;
    })
    .join("");
  planView.innerHTML = `
    <div class="meta">Goal: ${escapeHtml(plan.goal || "")}</div>
    ${gates}
    <div class="meta">Response: ${escapeHtml(JSON.stringify(plan.response || {}))}</div>`;
}

function displayGateParams(gate) {
  const params = { ...(gate.params || {}) };
  if (gate.type === "local_realtime_transcription") {
    params.buffer_ms = Math.min(Number(params.buffer_ms ?? 2000) || 2000, 2000);
    params.sample_interval_ms = Math.max(250, Math.min(Number(params.sample_interval_ms ?? 1000) || 1000, 1000));
  }
  return params;
}

function renderPlanning() {
  planView.innerHTML = `
    <div class="planning">
      <span class="spinner"></span>
      <strong>Planning</strong>
    </div>
    <div class="meta">Building realtime gates and response behavior for this media.</div>`;
}

function addLog(event) {
  logs.push(event);
  logCount.textContent = String(logs.length);
  appendEvent(logsView, event, "log");
}

function addUserMessage(event) {
  messages.push(event);
  messageCount.textContent = String(messages.length);
  appendMessage(messagesView, event);
}

function appendMessage(container, event) {
  const row = document.createElement("div");
  const role = event.role || (event.type === "user_instruction" ? "user" : "assistant");
  row.className = `message-row ${role === "user" ? "from-user" : "from-assistant"}`;
  const canSeek = event.stream_time_ms !== null && event.stream_time_ms !== undefined;
  const streamMeta = canSeek ? ` · stream=${formatMs(event.stream_time_ms)}` : "";
  row.innerHTML = `
    <div class="message-bubble">
      <div class="message-meta">wall=${formatMs(event.wall_ms)}${streamMeta}</div>
      <div class="message-content">${escapeHtml(event.message || event.text || "")}</div>
      ${canSeek ? `<button class="go-btn message-go">Go to ${formatMs(event.stream_time_ms)}</button>` : ""}
    </div>`;
  const button = row.querySelector("button");
  if (button) {
    button.addEventListener("click", () => showFramePreview(event.stream_time_ms));
  }
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function appendEvent(container, event, kind) {
  const row = document.createElement("div");
  row.className = `event ${event.bold ? "log-bold" : ""}`;
  const canSeek = event.stream_time_ms !== null && event.stream_time_ms !== undefined;
  row.innerHTML = `
    <div class="event-title">
      <strong>${formatMs(event.wall_ms)} wall</strong>
      <button class="go-btn" ${canSeek ? "" : "disabled"}>Go to ${formatMs(event.stream_time_ms)}</button>
    </div>
    <div class="meta">stream=${formatMs(event.stream_time_ms)}${event.trigger_gate_id ? ` · ${escapeHtml(event.trigger_gate_id)}` : ""}</div>
    <div class="event-body">${escapeHtml(event.message || event.text || "")}</div>`;
  const button = row.querySelector("button");
  button.addEventListener("click", () => {
    if (canSeek) showFramePreview(event.stream_time_ms);
  });
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

// What the current plan was built for. Replaying only skips planning while
// both still match, since either one changes the funnel.
let plannedPrompt = null;
let plannedMediaId = null;

// Set when a replay was triggered by the user pressing play, so playback
// resumes by itself once the server is ready for it.
let replayShouldAutoPlay = false;

function canReusePlan() {
  return Boolean(
    sessionId &&
      planCompleted &&
      ws &&
      ws.readyState === WebSocket.OPEN &&
      plannedPrompt === promptInput.value &&
      plannedMediaId === currentMediaId()
  );
}

function syncStartButton() {
  startBtn.textContent = canReusePlan() ? "Run again" : "Plan";
}

function clearRunViews() {
  logsView.innerHTML = "";
  messagesView.innerHTML = "";
  messages = [];
  logs = [];
  messageCount.textContent = "0";
  logCount.textContent = "0";
  playRequested = false;
  backendStarted = false;
  backendStarting = false;
  processingComplete = false;
}

function replaySession({ autoPlay = false } = {}) {
  // The funnel is already compiled; ask the server to run it over the media
  // again rather than paying for planning a second time.
  startBtn.disabled = true;
  clearRunViews();
  pauseAndResetActiveMedia();
  replayShouldAutoPlay = autoPlay;
  sessionState.textContent = "restarting";
  ws.send(JSON.stringify({ type: "restart" }));
}

async function startSession() {
  if (!currentMediaId()) return;
  if (canReusePlan()) {
    replaySession();
    return;
  }
  startBtn.disabled = true;
  planningInProgress = true;
  sessionState.textContent = "Planning";
  renderPlanning();
  pauseAndResetActiveMedia();
  setPlanningPlaybackLocked(true);
  clearRunViews();
  planCompleted = false;
  setPlanningPlaybackLocked(true);
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // Only the choices the user actually makes here. Everything else comes
    // from the server's StartRequest defaults, so the UI and the CLI cannot
    // drift apart.
    body: JSON.stringify({
      media: currentMediaId(),
      prompt: promptInput.value,
      speak: ttsToggle.checked
    })
  });
  if (!response.ok) {
    sessionState.textContent = "error";
    startBtn.disabled = false;
    planningInProgress = false;
    setPlanningPlaybackLocked(false);
    addLog({ wall_ms: 0, stream_time_ms: null, message: await response.text() });
    return;
  }
  const data = await response.json();
  sessionId = data.session_id;
  mediaInfo = data.media_info || mediaInfo;
  if (data.live) {
    await setupLiveMedia(mediaInfo);
  } else {
    setupMedia(data.media_url, data.audio_only, mediaInfo);
  }
  setPlanningPlaybackLocked(true);
  connectWebSocket(sessionId);
}

function setupMedia(url, audioOnly, info = null) {
  stopLivePreview();
  videoPlayer.pause();
  audioPlayer.pause();
  mediaInfo = info;
  activeLive = false;
  resetLiveClock();
  activeAudioOnly = audioOnly;
  videoPlayer.hidden = audioOnly;
  audioPlayer.hidden = true;
  audioSurface.hidden = !audioOnly;
  audioFileName.textContent = currentMediaLabel();
  activePlayer = audioOnly ? audioPlayer : videoPlayer;
  activePlayer.src = url;
  activePlayer.currentTime = 0;
  activePlayer.load();
  backendStarted = false;
  backendStarting = false;
  playRequested = false;
  protectedMediaTime = 0;
  restoringProtectedTime = false;
  processingComplete = false;
  setMainControlsLocked(false);
  previewPanel.hidden = true;
}

async function setupLiveMedia(info = null) {
  audioPlayer.pause();
  stopLivePreview();
  mediaInfo = info;
  activeLive = true;
  activeAudioOnly = false;
  videoPlayer.hidden = false;
  audioPlayer.hidden = true;
  audioSurface.hidden = true;
  activePlayer = videoPlayer;
  videoPlayer.controls = false;
  videoPlayer.muted = true;
  videoPlayer.playsInline = true;
  videoPlayer.removeAttribute("src");
  try {
    livePreviewStream = await openLivePreviewStream();
    videoPlayer.srcObject = livePreviewStream;
    await videoPlayer.play();
  } catch (error) {
    sessionState.textContent = "live source error";
    addLog({ wall_ms: currentMediaMs(), stream_time_ms: null, message: `Could not open ${currentMediaLabel().toLowerCase()}: ${error.message || error}` });
  }
  backendStarted = false;
  backendStarting = false;
  playRequested = false;
  protectedMediaTime = 0;
  restoringProtectedTime = false;
  processingComplete = false;
  resetLiveClock();
  setMainControlsLocked(false);
  previewPanel.hidden = true;
}

async function openLivePreviewStream() {
  if (currentMediaId() === LIVE_SCREEN_ID) {
    return openScreenShareStream();
  }
  return openCameraStream();
}

async function openCameraStream() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    addLog({ wall_ms: currentMediaMs(), stream_time_ms: null, message: "Live preview opened with camera and microphone." });
    return stream;
  } catch (error) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    addLog({
      wall_ms: currentMediaMs(),
      stream_time_ms: null,
      message: `Live preview opened with camera only; microphone preview was unavailable: ${error.message || error}`,
    });
    return stream;
  }
}

async function openScreenShareStream() {
  const getDisplayMedia = navigator.mediaDevices?.getDisplayMedia?.bind(navigator.mediaDevices);
  if (!getDisplayMedia) {
    throw new Error("Screen sharing is not supported by this browser.");
  }
  try {
    const stream = await getDisplayMedia({ video: true, audio: true });
    addLog({ wall_ms: currentMediaMs(), stream_time_ms: null, message: "Live screen share opened with optional shared audio." });
    attachLiveTrackEndedHandlers(stream);
    return stream;
  } catch (error) {
    const stream = await getDisplayMedia({ video: true, audio: false });
    addLog({
      wall_ms: currentMediaMs(),
      stream_time_ms: null,
      message: `Live screen share opened without audio: ${error.message || error}`,
    });
    attachLiveTrackEndedHandlers(stream);
    return stream;
  }
}

function attachLiveTrackEndedHandlers(stream) {
  for (const track of stream.getVideoTracks()) {
    track.addEventListener("ended", () => {
      stopLiveFramePump();
      stopLiveAudioPump();
      if (ws && ws.readyState === WebSocket.OPEN && backendStarted && !processingComplete) {
        ws.send(JSON.stringify({ type: "pause" }));
      }
      sessionState.textContent = "live source ended";
    });
  }
}

function stopLivePreview() {
  stopLiveAudioPump();
  stopLiveFramePump();
  if (livePreviewStream) {
    for (const track of livePreviewStream.getTracks()) track.stop();
  }
  livePreviewStream = null;
  if (videoPlayer.srcObject) videoPlayer.srcObject = null;
}

function connectWebSocket(id) {
  if (ws) {
    ws.close();
    ws = null;
  }
  stopLiveFramePump();
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${window.location.host}/ws/${id}`);
  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({ type: "plan" }));
  });
  ws.addEventListener("message", (message) => {
    const event = JSON.parse(message.data);
    if (event.type === "planning") {
      sessionState.textContent = "Planning";
      renderPlanning();
    }
    if (event.type === "plan") renderPlan(event.plan);
    if (event.type === "plan_completed") {
      planCompleted = true;
      planningInProgress = false;
      plannedPrompt = promptInput.value;
      plannedMediaId = currentMediaId();
      renderPlan(event.plan);
      sessionState.textContent = "Plan completed";
      startBtn.disabled = false;
      setPlanningPlaybackLocked(false);
      syncStartButton();
      if (activeLive) playRequested = true;
      maybeStartBackend();
    }
    if (event.type === "restarted") {
      renderPlan(event.plan);
      sessionState.textContent = "ready to replay";
      startBtn.disabled = false;
      setMainControlsLocked(false);
      setPlanningPlaybackLocked(false);
      syncStartButton();
      if (activeLive) {
        playRequested = true;
        maybeStartBackend();
      } else if (replayShouldAutoPlay) {
        replayShouldAutoPlay = false;
        activePlayer.play().catch(() => {});
      }
    }
    if (event.type === "started") {
      backendStarted = true;
      backendStarting = false;
      sessionState.textContent = "running";
      setMainControlsLocked(true);
      if (activeLive) {
        startLiveClock();
        startLiveMediaPumps();
      }
      if (playRequested && activePlayer.paused) {
        activePlayer.play().catch(() => {});
      }
    }
    if (event.type === "log") addLog(event);
    if (event.type === "user_message") addUserMessage(event);
    if (event.type === "user_instruction") addUserMessage({ ...event, role: "user" });
    if (event.type === "audio") playGeneratedAudio(event);
    if (event.type === "done") {
      processingComplete = true;
      sessionState.textContent = "complete";
      stopLiveFramePump();
      stopLiveAudioPump();
      setMainControlsLocked(false);
      syncStartButton();
    }
    if (event.type === "error") {
      sessionState.textContent = "error";
      planningInProgress = false;
      setPlanningPlaybackLocked(false);
      stopLiveFramePump();
      stopLiveAudioPump();
      addLog({ wall_ms: event.wall_ms ?? currentMediaMs(), stream_time_ms: event.stream_time_ms ?? null, message: event.message });
    }
  });
  ws.addEventListener("close", () => {
    if (sessionState.textContent === "running") sessionState.textContent = "disconnected";
    startBtn.disabled = false;
  });
}

function maybeStartBackend() {
  if (!planCompleted || !playRequested || backendStarted || backendStarting || !ws || ws.readyState !== WebSocket.OPEN) return;
  backendStarting = true;
  sessionState.textContent = "starting";
  setMainControlsLocked(true);
  ws.send(JSON.stringify({ type: "start" }));
}

function showFramePreview(ms) {
  const timeMs = Math.max(0, Math.floor(Number(ms)));
  if (!Number.isFinite(timeMs) || !sessionId) return;
  previewPanel.hidden = false;
  previewTime.textContent = formatMs(timeMs);
  if (activeLive) {
    previewImage.removeAttribute("src");
    previewImage.alt = "Live streams do not support frame preview rewind";
    return;
  }
  if (currentIsAudioOnly()) {
    previewImage.removeAttribute("src");
    previewImage.alt = "Audio-only media has no video frame preview";
    return;
  }
  previewImage.alt = `Frame at ${formatMs(timeMs)}`;
  previewImage.src = `/api/sessions/${sessionId}/frame?time_ms=${timeMs}&cache=${Date.now()}`;
}

function playGeneratedAudio(event) {
  if (!event?.url) return;
  if (currentGeneratedAudio) {
    currentGeneratedAudio.pause();
    currentGeneratedAudio.removeAttribute("src");
    currentGeneratedAudio.load();
  }
  const audio = new Audio(event.url);
  currentGeneratedAudio = audio;
  audio.preload = "auto";
  const clearCurrent = () => {
    if (currentGeneratedAudio === audio) currentGeneratedAudio = null;
  };
  audio.addEventListener("ended", clearCurrent, { once: true });
  audio.addEventListener("error", clearCurrent, { once: true });
  audio.play().catch(clearCurrent);
}

function sendLivePrompt() {
  const prompt = livePromptInput.value.trim();
  if (!prompt || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "prompt", prompt }));
  livePromptInput.value = "";
}

function handleMediaPlay() {
  if (activeLive && (!sessionId || !planCompleted)) {
    return;
  }
  if (planningInProgress || (sessionId && !planCompleted)) {
    activePlayer.pause();
    return;
  }
  if (processingComplete) {
    // The run is over. Playing the media again should watch it again rather
    // than scrub video with nothing looking at it; the funnel is already
    // compiled, so this costs no planning.
    if (canReusePlan()) {
      replaySession({ autoPlay: true });
    }
    return;
  }
  playRequested = true;
  if (!backendStarted) {
    activePlayer.pause();
    maybeStartBackend();
    return;
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    if (activeLive) {
      startLiveClock();
      startLiveMediaPumps();
    }
    ws.send(JSON.stringify({ type: "resume" }));
  }
}

function handleMediaPause() {
  if (activeLive && !backendStarted) return;
  if (backendStarted && !processingComplete && !activePlayer.ended && ws && ws.readyState === WebSocket.OPEN) {
    if (activeLive) {
      pauseLiveClock();
      stopLiveFramePump();
      stopLiveAudioPump();
    }
    ws.send(JSON.stringify({ type: "pause" }));
  }
}

function protectLiveTimeline(event) {
  const player = event.currentTarget;
  if (activeLive) return;
  if (player !== activePlayer || !backendStarted || processingComplete || restoringProtectedTime) return;
  const currentTime = player.currentTime || 0;
  if (currentTime < protectedMediaTime - 0.25 || currentTime > protectedMediaTime + 0.75) {
    restoringProtectedTime = true;
    player.currentTime = protectedMediaTime;
    player.addEventListener(
      "seeked",
      () => {
        restoringProtectedTime = false;
      },
      { once: true }
    );
  }
}

function recordLiveTime(event) {
  const player = event.currentTarget;
  if (activeLive) return;
  if (player !== activePlayer || restoringProtectedTime) return;
  if (backendStarted && !processingComplete) {
    protectedMediaTime = Math.max(protectedMediaTime, player.currentTime || 0);
  }
}

function setMainControlsLocked(locked) {
  videoPlayer.controls = !locked && !activeAudioOnly && !activeLive;
  audioPlayer.controls = false;
  livePlayPauseBtn.hidden = activeAudioOnly || (!locked && !activeLive);
  audioSurface.hidden = !activeAudioOnly;
  audioPlayBtn.disabled = planningInProgress || (sessionId && !planCompleted);
}

function setPlanningPlaybackLocked(locked) {
  if (!locked) {
    if (!backendStarted) setMainControlsLocked(false);
    return;
  }
  videoPlayer.controls = false;
  audioPlayer.controls = false;
  livePlayPauseBtn.hidden = true;
  if (activeAudioOnly) {
    audioSurface.hidden = false;
    audioPlayBtn.disabled = true;
  }
}

function pauseAndResetActiveMedia() {
  if (activeLive) {
    resetLiveClock();
    return;
  }
  activePlayer.pause();
  try {
    activePlayer.currentTime = 0;
  } catch {
    activePlayer.addEventListener(
      "loadedmetadata",
      () => {
        activePlayer.currentTime = 0;
      },
      { once: true }
    );
  }
  protectedMediaTime = 0;
}

function toggleLivePlayback() {
  if (planningInProgress || (sessionId && !planCompleted)) return;
  if (activePlayer.paused) {
    playRequested = true;
    if (!backendStarted && !processingComplete) {
      maybeStartBackend();
      return;
    }
    if (activeLive) startLiveClock();
    if (activeLive) startLiveMediaPumps();
    activePlayer.play().catch(() => {});
  } else {
    if (activeLive) stopLiveFramePump();
    if (activeLive) stopLiveAudioPump();
    activePlayer.pause();
  }
}

function startLiveMediaPumps() {
  startLiveFramePump();
  startLiveAudioPump();
}

function startLiveFramePump() {
  if (!activeLive || liveFrameTimer || !ws || ws.readyState !== WebSocket.OPEN) return;
  liveFrameCanvas = liveFrameCanvas || document.createElement("canvas");
  const sendFrame = () => {
    if (!activeLive || !backendStarted || videoPlayer.paused || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (!videoPlayer.videoWidth || !videoPlayer.videoHeight) return;
    const scale = Math.min(1, LIVE_FRAME_MAX_WIDTH / videoPlayer.videoWidth);
    const width = Math.max(1, Math.round(videoPlayer.videoWidth * scale));
    const height = Math.max(1, Math.round(videoPlayer.videoHeight * scale));
    liveFrameCanvas.width = width;
    liveFrameCanvas.height = height;
    const context = liveFrameCanvas.getContext("2d", { alpha: false });
    context.drawImage(videoPlayer, 0, 0, width, height);
    ws.send(JSON.stringify({
      type: "live_frame",
      stream_time_ms: currentMediaMs(),
      sequence_id: liveFrameSequence++,
      image: liveFrameCanvas.toDataURL("image/jpeg", 0.65),
    }));
  };
  sendFrame();
  liveFrameTimer = window.setInterval(sendFrame, LIVE_FRAME_INTERVAL_MS);
}

function stopLiveFramePump() {
  if (liveFrameTimer) {
    window.clearInterval(liveFrameTimer);
    liveFrameTimer = null;
  }
}

async function startLiveAudioPump() {
  if (!activeLive || liveAudioProcessor || !ws || ws.readyState !== WebSocket.OPEN) return;
  if (!livePreviewStream || !livePreviewStream.getAudioTracks().length) {
    addLog({ wall_ms: currentMediaMs(), stream_time_ms: null, message: "Live microphone is unavailable; no browser audio will be sent." });
    return;
  }
  try {
    liveAudioContext = liveAudioContext || new (window.AudioContext || window.webkitAudioContext)();
    if (liveAudioContext.state === "suspended") await liveAudioContext.resume();
    liveAudioSourceNode = liveAudioContext.createMediaStreamSource(livePreviewStream);
    liveAudioProcessor = liveAudioContext.createScriptProcessor(4096, 1, 1);
    liveAudioProcessor.onaudioprocess = (event) => {
      if (!activeLive || !backendStarted || videoPlayer.paused || !ws || ws.readyState !== WebSocket.OPEN) return;
      const input = event.inputBuffer.getChannelData(0);
      const downsampled = downsampleAudio(input, liveAudioContext.sampleRate, LIVE_AUDIO_SAMPLE_RATE);
      enqueueLiveAudio(downsampled);
    };
    liveAudioSourceNode.connect(liveAudioProcessor);
    liveAudioProcessor.connect(liveAudioContext.destination);
    addLog({ wall_ms: currentMediaMs(), stream_time_ms: null, message: "Live microphone audio pump started." });
  } catch (error) {
    addLog({ wall_ms: currentMediaMs(), stream_time_ms: null, message: `Could not start live microphone audio pump: ${error.message || error}` });
    stopLiveAudioPump();
  }
}

function stopLiveAudioPump() {
  if (liveAudioProcessor) {
    liveAudioProcessor.disconnect();
    liveAudioProcessor.onaudioprocess = null;
  }
  if (liveAudioSourceNode) liveAudioSourceNode.disconnect();
  liveAudioProcessor = null;
  liveAudioSourceNode = null;
  liveAudioChunks = [];
  liveAudioBufferedSamples = 0;
}

function enqueueLiveAudio(samples) {
  if (!samples.length) return;
  liveAudioChunks.push(samples);
  liveAudioBufferedSamples += samples.length;
  const samplesPerChunk = Math.max(1, Math.round((LIVE_AUDIO_SAMPLE_RATE * LIVE_AUDIO_CHUNK_MS) / 1000));
  while (liveAudioBufferedSamples >= samplesPerChunk) {
    const chunk = new Float32Array(samplesPerChunk);
    let offset = 0;
    while (offset < samplesPerChunk && liveAudioChunks.length) {
      const head = liveAudioChunks[0];
      const take = Math.min(head.length, samplesPerChunk - offset);
      chunk.set(head.subarray(0, take), offset);
      offset += take;
      if (take === head.length) {
        liveAudioChunks.shift();
      } else {
        liveAudioChunks[0] = head.subarray(take);
      }
    }
    liveAudioBufferedSamples -= samplesPerChunk;
    ws.send(JSON.stringify({
      type: "live_audio",
      stream_time_ms: currentMediaMs(),
      sequence_id: liveAudioSequence++,
      sample_rate: LIVE_AUDIO_SAMPLE_RATE,
      chunk_ms: LIVE_AUDIO_CHUNK_MS,
      samples: int16AudioToBase64(chunk),
    }));
  }
}

function downsampleAudio(input, sourceRate, targetRate) {
  if (targetRate >= sourceRate) return new Float32Array(input);
  const ratio = sourceRate / targetRate;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += input[j];
    output[i] = sum / Math.max(1, end - start);
  }
  return output;
}

function int16AudioToBase64(samples) {
  const bytes = new Uint8Array(samples.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
  }
  let binary = "";
  const block = 0x8000;
  for (let i = 0; i < bytes.length; i += block) {
    binary += String.fromCharCode(...bytes.subarray(i, i + block));
  }
  return btoa(binary);
}

function startLiveClock() {
  if (liveRunningSinceMs === null) {
    liveRunningSinceMs = performance.now();
  }
}

function pauseLiveClock() {
  if (liveRunningSinceMs !== null) {
    liveElapsedBeforePauseMs += performance.now() - liveRunningSinceMs;
    liveRunningSinceMs = null;
  }
}

function resetLiveClock() {
  liveRunningSinceMs = null;
  liveElapsedBeforePauseMs = 0;
  liveFrameSequence = 0;
  liveAudioSequence = 0;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function isLiveMedia(path) {
  return path === LIVE_CAMERA_ID || path === LIVE_SCREEN_ID;
}

function liveMediaLabel(path) {
  if (path === LIVE_CAMERA_ID) return "Live camera";
  if (path === LIVE_SCREEN_ID) return "Live screen / website";
  return path;
}

function clampDisplayedMediaMs(ms) {
  const maxVideoMs = Number(mediaInfo?.video_duration_ms);
  if (!currentIsAudioOnly() && Number.isFinite(maxVideoMs) && maxVideoMs > 0) {
    return Math.min(ms, maxVideoMs);
  }
  return ms;
}

startBtn.addEventListener("click", startSession);
sourceSelect.addEventListener("change", onSourceChange);
fileInput.addEventListener("change", onFileChosen);
promptInput.addEventListener("input", syncStartButton);
sendPromptBtn.addEventListener("click", sendLivePrompt);
livePlayPauseBtn.addEventListener("click", toggleLivePlayback);
audioPlayBtn.addEventListener("click", toggleLivePlayback);
livePromptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") sendLivePrompt();
});
for (const player of [videoPlayer, audioPlayer]) {
  player.addEventListener("play", handleMediaPlay);
  player.addEventListener("pause", handleMediaPause);
  player.addEventListener("seeking", protectLiveTimeline);
  player.addEventListener("timeupdate", recordLiveTime);
}

renderPlan(null);
syncSourceControls();
sessionState.textContent = "choose a file";
updateClock();
