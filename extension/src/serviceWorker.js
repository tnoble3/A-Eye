const API_URL = "http://localhost:8000/analyze";
const CONTEXT_MENU_ID = "aeye_check_image";
const SETTINGS_KEY = "aeyeSettings";
const RESULT_KEY = "aeyeAnalysisState";
const DEFAULT_SETTINGS = Object.freeze({
  persistResults: false,
  redactSourceUrl: true,
});

function buildSourceSnapshot(srcUrl, redactSourceUrl) {
  try {
    const url = new URL(srcUrl);
    const fileName = url.pathname.split("/").filter(Boolean).pop() || "selected-image";
    return {
      host: url.host,
      pathLabel: `${url.host} / ${fileName}`,
      url: redactSourceUrl ? null : srcUrl,
    };
  } catch {
    return {
      host: "unknown-host",
      pathLabel: redactSourceUrl ? "selected-image" : srcUrl,
      url: redactSourceUrl ? null : srcUrl,
    };
  }
}

async function ensureContextMenu() {
  await chrome.contextMenus.removeAll();
  chrome.contextMenus.create({
    id: CONTEXT_MENU_ID,
    title: "Analyze image with A-Eye",
    contexts: ["image"],
  });
}

async function getSettings() {
  const stored = await chrome.storage.local.get(SETTINGS_KEY);
  return {
    ...DEFAULT_SETTINGS,
    ...(stored[SETTINGS_KEY] || {}),
  };
}

async function getAnalysisState() {
  const sessionState = await chrome.storage.session.get(RESULT_KEY);
  if (sessionState[RESULT_KEY]) {
    return sessionState[RESULT_KEY];
  }

  const persistedState = await chrome.storage.local.get(RESULT_KEY);
  return persistedState[RESULT_KEY] || null;
}

async function setAnalysisState(state, settings) {
  await chrome.storage.session.set({ [RESULT_KEY]: state });
  if (settings.persistResults && state.status !== "running") {
    await chrome.storage.local.set({ [RESULT_KEY]: state });
  } else {
    await chrome.storage.local.remove(RESULT_KEY);
  }
}

async function clearAnalysisState() {
  await chrome.storage.session.remove(RESULT_KEY);
  await chrome.storage.local.remove(RESULT_KEY);
}

async function updateBadgeForState(state) {
  if (!state) {
    await chrome.action.setBadgeText({ text: "" });
    await chrome.action.setTitle({ title: "A-Eye" });
    return;
  }

  if (state.status === "running") {
    await chrome.action.setBadgeBackgroundColor({ color: "#6a5d2f" });
    await chrome.action.setBadgeText({ text: "..." });
    await chrome.action.setTitle({ title: "A-Eye: analysis running" });
    return;
  }

  if (state.status === "error") {
    await chrome.action.setBadgeBackgroundColor({ color: "#8a3f34" });
    await chrome.action.setBadgeText({ text: "!" });
    await chrome.action.setTitle({ title: "A-Eye: last analysis failed" });
    return;
  }

  const finalConfidence = state.result?.final_confidence ?? 0;
  const badgeText = String(Math.round(finalConfidence * 100)).slice(0, 3);
  await chrome.action.setBadgeBackgroundColor({ color: "#2d6653" });
  await chrome.action.setBadgeText({ text: badgeText });
  await chrome.action.setTitle({
    title: `A-Eye: ${badgeText}% AI-generation likelihood`,
  });
}

async function analyzeImageFromContextMenu(srcUrl) {
  const settings = await getSettings();
  const source = buildSourceSnapshot(srcUrl, settings.redactSourceUrl);
  const startedAt = new Date().toISOString();

  await setAnalysisState(
    {
      status: "running",
      source,
      requestedAt: startedAt,
      privacy: {
        persistResults: settings.persistResults,
        redactSourceUrl: settings.redactSourceUrl,
      },
    },
    settings,
  );
  await updateBadgeForState({ status: "running" });

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image_url: srcUrl }),
      cache: "no-store",
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.detail || `Request failed with ${response.status}`);
    }

    const data = await response.json();
    const state = {
      status: "complete",
      source,
      requestedAt: startedAt,
      completedAt: new Date().toISOString(),
      privacy: {
        persistResults: settings.persistResults,
        redactSourceUrl: settings.redactSourceUrl,
      },
      result: data,
    };
    await setAnalysisState(state, settings);
    await updateBadgeForState(state);
  } catch (error) {
    const state = {
      status: "error",
      source,
      requestedAt: startedAt,
      completedAt: new Date().toISOString(),
      privacy: {
        persistResults: settings.persistResults,
        redactSourceUrl: settings.redactSourceUrl,
      },
      error: String(error),
    };
    await setAnalysisState(state, settings);
    await updateBadgeForState(state);
  }
}

chrome.runtime.onInstalled.addListener(() => {
  void ensureContextMenu();
});

chrome.runtime.onStartup.addListener(() => {
  void ensureContextMenu();
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId !== CONTEXT_MENU_ID || !info.srcUrl) {
    return;
  }

  void analyzeImageFromContextMenu(info.srcUrl);
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "get-popup-state") {
    (async () => {
      const [settings, analysisState] = await Promise.all([
        getSettings(),
        getAnalysisState(),
      ]);
      sendResponse({ settings, analysisState });
    })();
    return true;
  }

  if (message?.type === "save-settings") {
    (async () => {
      const nextSettings = {
        ...DEFAULT_SETTINGS,
        ...(message.settings || {}),
      };
      await chrome.storage.local.set({ [SETTINGS_KEY]: nextSettings });

      // Turning persistence off should also drop any durable copy of the last result.
      if (!nextSettings.persistResults) {
        await chrome.storage.local.remove(RESULT_KEY);
      } else {
        const sessionState = await chrome.storage.session.get(RESULT_KEY);
        if (sessionState[RESULT_KEY] && sessionState[RESULT_KEY].status !== "running") {
          await chrome.storage.local.set({ [RESULT_KEY]: sessionState[RESULT_KEY] });
        }
      }

      sendResponse({ ok: true });
    })();
    return true;
  }

  if (message?.type === "clear-analysis-state") {
    (async () => {
      await clearAnalysisState();
      await updateBadgeForState(null);
      sendResponse({ ok: true });
    })();
    return true;
  }

  return false;
});
