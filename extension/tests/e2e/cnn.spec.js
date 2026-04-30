const { test, expect } = require("@playwright/test");
const { launchExtension, setExtensionState } = require("./helpers/extension");

test("popup shows CNN as unavailable when the backend does not return a score", async () => {
  const app = await launchExtension();

  try {
    await setExtensionState(app.serviceWorker, {
      settings: {
        persistResults: false,
        redactSourceUrl: false,
      },
      analysisState: {
        status: "complete",
        source: {
          host: "images.example.com",
          pathLabel: "images.example.com / cnn-missing.png",
          url: "https://images.example.com/cnn-missing.png",
        },
        requestedAt: "2026-04-26T15:10:00.000Z",
        completedAt: "2026-04-26T15:10:04.000Z",
        privacy: {
          persistResults: false,
          redactSourceUrl: false,
        },
        result: {
          final_confidence: 0.58,
          feature_confidence: 0.58,
          cnn_confidence: null,
          signals: [
            {
              name: "cnn_baseline_unavailable",
              score: 0.58,
              detail: "CNN checkpoint not found at /app/ml/artifacts/cnn_baseline/best_model.pt.",
            },
          ],
          meta: {
            deployed_cnn: false,
            privacy: {
              processing_mode: "stateless",
            },
          },
        },
      },
    });

    const page = await app.context.newPage();
    await page.goto(`chrome-extension://${app.extensionId}/src/ui/popup.html`);

    await expect(page.locator("#summary")).toContainText("Feature First mode");
    await expect(page.locator("#summary")).toContainText("CNN Model Confidence");
    await expect(page.locator("#summary")).toContainText("N/A");
    await expect(page.locator("#summary")).toContainText("Unavailable");
    await expect(page.locator("#signals")).toContainText("cnn_baseline_unavailable (58%)");
  } finally {
    await app.cleanup();
  }
});
