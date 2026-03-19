const API_URL = "https://localhost:8000/analyze";

chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "aeye_check_image",
        title: "Check image with Authenticity",
        contexts: ["image"],
    });
});

chrome.contextMenus.onClicked.addListener(async(info) => {
    if (info.menuItemID !== "aeye_check_image") return;
    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"},
            body: JSON.stringify({ imageUrl: info.srcUrl }),
        });
        const data = await res.json();
    //store latest result for popup to read
    await chrome.storage.local.set({ lastResult: data });

    //otional: show a quick notification via badge text
    chrome.action.setBadgeText({ text: "✓" });
  } catch (err) {
    await chrome.storage.local.set({ lastResult: { error: String(err) } });
    chrome.action.setBadgeText({ text: "!" });
  }
});