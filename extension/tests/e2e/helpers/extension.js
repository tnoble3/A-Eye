const fs = require("fs");
const os = require("os");
const path = require("path");
const { chromium } = require("@playwright/test");

const SETTINGS_KEY = "aeyeSettings";
const RESULT_KEY = "aeyeAnalysisState";

async function launchExtension() {
  const extensionPath = path.resolve(__dirname, "../../..");
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "aeye-playwright-"));
  const context = await chromium.launchPersistentContext(userDataDir, {
    channel: "chromium",
    headless: false,
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
    ],
  });

  let serviceWorker = context.serviceWorkers()[0];
  if (!serviceWorker) {
    serviceWorker = await context.waitForEvent("serviceworker");
  }

  return {
    context,
    extensionId: new URL(serviceWorker.url()).host,
    serviceWorker,
    async cleanup() {
      await context.close();
      fs.rmSync(userDataDir, { recursive: true, force: true });
    },
  };
}

async function setExtensionState(serviceWorker, { settings, analysisState }) {
  await serviceWorker.evaluate(
    async ({ settingsKey, resultKey, settings, analysisState }) => {
      await chrome.storage.local.clear();
      await chrome.storage.session.clear();

      if (settings) {
        await chrome.storage.local.set({ [settingsKey]: settings });
      }

      if (analysisState) {
        await chrome.storage.session.set({ [resultKey]: analysisState });
      }
    },
    {
      settingsKey: SETTINGS_KEY,
      resultKey: RESULT_KEY,
      settings,
      analysisState,
    },
  );
}

async function getStoredAnalysisState(serviceWorker) {
  return serviceWorker.evaluate(async ({ resultKey }) => {
    const sessionState = await chrome.storage.session.get(resultKey);
    const localState = await chrome.storage.local.get(resultKey);
    return sessionState[resultKey] ?? localState[resultKey] ?? null;
  }, { resultKey: RESULT_KEY });
}

module.exports = {
  launchExtension,
  setExtensionState,
  getStoredAnalysisState,
};
