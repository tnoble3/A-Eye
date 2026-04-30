const { test, expect } = require("@playwright/test");
const {
  launchExtension,
  setExtensionState,
  getStoredAnalysisState,
} = require("./helpers/extension");

async function seedPopupState(serviceWorker) {
  await setExtensionState(serviceWorker, {
    settings: {
      persistResults: false,
      redactSourceUrl: false,
    },
    analysisState: {
      status: "complete",
      source: {
        host: "images.example.com",
        pathLabel: "images.example.com / sample.png",
        url: "https://images.example.com/sample.png",
      },
      requestedAt: "2026-04-26T15:00:00.000Z",
      completedAt: "2026-04-26T15:00:05.000Z",
      privacy: {
        persistResults: false,
        redactSourceUrl: false,
      },
      result: {
        final_confidence: 0.72,
        feature_confidence: 0.81,
        cnn_confidence: 0.66,
        signals: [
          {
            name: "compression_anomalies",
            score: 0.12,
            detail:
              "Compression anomalies are limited. ELA did not find strong localized recompression differences.",
          },
        ],
        meta: {
          deployed_cnn: true,
          privacy: {
            processing_mode: "stateless",
          },
        },
      },
    },
  });
}

test("popup renders stored analysis state and clears it on demand", async () => {
  const app = await launchExtension();

  try {
    await seedPopupState(app.serviceWorker);

    const page = await app.context.newPage();
    await page.goto(`chrome-extension://${app.extensionId}/src/ui/popup.html`);

    await expect(page.locator("#statusPill")).toHaveText("Ready");
    await expect(page.locator("#summary")).toContainText("Estimated AI Generation Likelihood");
    await expect(page.locator("#summary")).toContainText("72%");
    await expect(page.locator("#summary")).toContainText("CNN Model Confidence");
    await expect(page.locator("#summary")).toContainText("66%");
    await expect(page.locator("#sourceCard")).toContainText("images.example.com / sample.png");
    await expect(page.locator("#signals")).toContainText("compression_anomalies (12%)");

    await page.getByRole("button", { name: "Clear Result" }).click();

    await expect(page.locator("#statusPill")).toHaveText("Idle");
    await expect(page.locator("#summary")).toContainText(
      'Use the "Analyze image with A-Eye" context menu entry on any image to start a scan.',
    );
    await expect(page.locator("#sourceCard")).toContainText(
      "No image has been analyzed in this browser session.",
    );

    await expect.poll(() => getStoredAnalysisState(app.serviceWorker)).toBeNull();
  } finally {
    await app.cleanup();
  }
});
