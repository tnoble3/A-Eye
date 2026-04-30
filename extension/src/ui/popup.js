
function createElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== undefined) {
    element.textContent = text;
  }
  return element;
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "N/A";
  }
  return `${Math.round(value * 100)}%`;
}

function appendScoreCard(root, { value, label, meta, wide = false }) {
  const className = wide ? "score-card score-card-wide" : "score-card";
  const card = createElement("div", className);
  card.appendChild(createElement("strong", "", value));
  card.appendChild(createElement("div", "muted", label));
  if (meta) {
    card.appendChild(createElement("div", "meta", meta));
  }
  root.appendChild(card);
}

//converts an ISO timestamp into a human friendly time format
function formatTimestamp(value) {
  if (!value) {
    return "unknown time";
  }

  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) {
    return value;
  }

  return timestamp.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

//here, the popup sends a message to the service worker to request the current state of the popup, 
// //which includes both the persisted settings and the latest analysis result. The service worker responds with this information, 
// allowing the popup to render the UI accordingly.
async function getPopupState() {
  return chrome.runtime.sendMessage({ type: "get-popup-state" });
}

//present the user with the option to save their privacy settings. They have the option to hide the full URL of the analyzed image and to 
//choose whether analysis results should persist across browser sessions.
//  When the user clicks the "Save Settings" button, the popup sends a message to the service worker with the updated settings, 
// which are then saved in storage and applied to future analyses.
async function saveSettings(settings) {
  await chrome.runtime.sendMessage({ type: "save-settings", settings });
}

async function clearAnalysisState() {
  await chrome.runtime.sendMessage({ type: "clear-analysis-state" });
}

//this shows the status of the analysis
function renderStatusPill(state) {
  const pill = document.getElementById("statusPill");
  pill.className = "status-pill";

  if (!state) {
    pill.classList.add("idle");
    pill.textContent = "Idle";
    return;
  }
  if (state.status === "running") {
    pill.classList.add("warn");
    pill.textContent = "Analyzing";
    return;
  }
  if (state.status === "error") {
    pill.classList.add("error");
    pill.textContent = "Error";
    return;
  }
  pill.textContent = "Ready";
}

function renderSummary(state) {
  const summary = document.getElementById("summary");
  summary.innerHTML = "";

  if (!state) {
    const muted = createElement(
      "div",
      "muted",
      'Use the "Analyze image with A-Eye" context menu entry on any image to start a scan.',
    );
    summary.appendChild(muted);
    return;
  }

  if (state.status === "running") {
    summary.appendChild(createElement("div", "big", "Working"));
    summary.appendChild(
      createElement(
        "div",
        "muted",
        `Analysis started ${formatTimestamp(state.requestedAt)}. The popup will refresh when the result lands.`,
      ),
    );
    return;
  }

  if (state.status === "error") {
    summary.appendChild(createElement("div", "big", "Request Failed"));
    summary.appendChild(createElement("div", "muted error", state.error || "Unknown error"));
    return;
  }

  const result = state.result || {};
  const modeLabel = result.meta?.deployed_cnn ? "Hybrid Mode" : "Feature First mode";
  const privacyLabel = result.meta?.privacy?.processing_mode || "stateless";
  const cnnAvailable = typeof result.cnn_confidence === "number";

  const scoreGrid = createElement("div", "score-grid");

  appendScoreCard(scoreGrid, {
    value: formatPercent(result.final_confidence),
    label: "Estimated AI Generation Likelihood",
    meta: modeLabel,
    wide: true,
  });
  appendScoreCard(scoreGrid, {
    value: formatPercent(result.feature_confidence),
    label: "Likelyhood Based on Feature Layer Confidence",
    meta: privacyLabel,
  });
  appendScoreCard(scoreGrid, {
    value: formatPercent(result.cnn_confidence),
    label: "Likelyhood Based on CNN Model Confidence",
    meta: cnnAvailable ? "Baseline CNN" : "Unavailable",
  });

  summary.appendChild(scoreGrid);
}

//the source card shows what image was analyzed. When URL redaction is enabled,
function renderSource(state) {
  const sourceCard = document.getElementById("sourceCard");
  sourceCard.innerHTML = "";

  if (!state?.source) {
    sourceCard.textContent = "No image has been analyzed in this browser session.";
    return;
  }

  const hostLabel = createElement("div", "info-label", "Selected Image");
  const value = createElement("div", "info-value", state.source.pathLabel || "selected-image");

  if (state.source.url) {
    const preview = createElement("img", "image-preview");
    preview.src = state.source.url;
    preview.alt = state.source.pathLabel || "Selected image preview";
    sourceCard.appendChild(preview);
  } else {
    sourceCard.appendChild(
      createElement("div", "info-value", "Preview hidden while source URL masking is enabled."),
    );
  }

  sourceCard.appendChild(hostLabel);
  sourceCard.appendChild(value);

  if (state.source.url) {
    sourceCard.appendChild(createElement("div", "info-label", "Full URL"));
    sourceCard.appendChild(createElement("div", "info-value", state.source.url));
  }
  sourceCard.appendChild(createElement("div", "info-label", "Last Request"));
  sourceCard.appendChild(
    createElement(
      "div",
      "info-value",
      state.completedAt ? formatTimestamp(state.completedAt) : formatTimestamp(state.requestedAt),
    ),
  );
}

//signals are the feature-level explanations returned by the backend. The popup
//then gives the user reasons behind the score presented.
function renderSignals(state) {
  const signalsRoot = document.getElementById("signals");
  signalsRoot.innerHTML = "";
  if (!state) {
    signalsRoot.textContent = "Run an analysis from the image context menu to populate feature-level findings.";
    return;
  }
  if (state.status === "running") {
    signalsRoot.textContent = "The backend is processing the current image.";
    return;
  }
  if (state.status === "error") {
    signalsRoot.appendChild(createElement("div", "sig error", state.error || "Unknown error"));
    return;
  }
  const result = state.result || {};
  const signals = result.signals || [];
  if (signals.length === 0) {
    signalsRoot.textContent = "No signals were returned by the backend.";
    return;
  }

  for (const signal of signals) {
    const card = createElement("div", "sig");
    card.appendChild(
      createElement(
        "div",
        "sig-title",
        `${signal.name} (${Math.round((signal.score || 0) * 100)}%)`,
      ),
    );
    card.appendChild(createElement("div", "muted", signal.detail || ""));
    signalsRoot.appendChild(card);
  }
}

function renderControls(settings) {
  document.getElementById("redactSourceUrl").checked = settings.redactSourceUrl;
  document.getElementById("persistResults").checked = settings.persistResults;
}
async function render() {
  const state = await getPopupState();
  const settings = state?.settings || {
    persistResults: false,
    redactSourceUrl: true,
  };
  const analysisState = state?.analysisState || null;
  renderStatusPill(analysisState);
  renderSummary(analysisState);
  renderSource(analysisState);
  renderSignals(analysisState);
  renderControls(settings);
}

async function main() {
  document.getElementById("saveSettings").addEventListener("click", async () => {
    await saveSettings({
      redactSourceUrl: document.getElementById("redactSourceUrl").checked,
      persistResults: document.getElementById("persistResults").checked,
    });
    await render();
  });

  document.getElementById("clearResult").addEventListener("click", async () => {
    await clearAnalysisState();
    await render();
  });

  chrome.storage.onChanged.addListener(() => {
    void render();
  });

  await render();
}

void main();
