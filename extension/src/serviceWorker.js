const API_URL = "http://localhost:8000/analyze";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "aeye_check_image",
    title: "Check image authenticity",
    contexts: ["image"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== "aeye_check_image" || !info.srcUrl) {
    return;
  }

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image_url: info.srcUrl }),
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.detail || `Request failed with ${response.status}`);
    }

    const data = await response.json();

    // The popup can read the last result from storage and display it immediately, just something to make it feel more responsive
    await chrome.storage.local.set({ lastResult: data });
    chrome.action.setBadgeBackgroundColor({ color: "#2d6653" });
    chrome.action.setBadgeText({ text: "OK" });
  } catch (error) {
    await chrome.storage.local.set({ lastResult: { error: String(error) } });
    chrome.action.setBadgeBackgroundColor({ color: "#8a3f34" });
    chrome.action.setBadgeText({ text: "!" });
  }
});
