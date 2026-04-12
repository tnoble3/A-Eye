async function main() {
  const el = document.getElementById("content");
  const { lastResult } = await chrome.storage.local.get("lastResult");

  if (!lastResult) {
    return;
  }

  if (lastResult.error) {
    el.innerHTML = `<div class="error">Error: ${lastResult.error}</div>`;
    return;
  }
//this area needs a bit more work. Leaving this for last, focusing on the core functionality first.
  const finalPercent = Math.round((lastResult.final_confidence || 0) * 100);
  const featurePercent = Math.round((lastResult.feature_confidence || 0) * 100);
  const modeLabel = lastResult.meta?.deployed_cnn ? "Hybrid mode" : "Feature-first stub";

  el.innerHTML = `
    <div class="big">${finalPercent}%</div>
    <div class="muted">Estimated AI-generation likelihood.</div>
    <div class="meta">${modeLabel}</div>
    <div class="signals">
      <div class="sig">
        <div class="sig-title">Feature Layer</div>
        <div class="muted">${featurePercent}% confidence from the current forensic proxy.</div>
      </div>
      ${(lastResult.signals || []).map((signal) => `
        <div class="sig">
          <div class="sig-title">${signal.name} (${Math.round(signal.score * 100)}%)</div>
          <div class="muted">${signal.detail}</div>
        </div>
      `).join("")}
    </div>
  `;
}

main();
