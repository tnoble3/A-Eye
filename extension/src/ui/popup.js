async function main() {
    const el = document.getElementById("content");
    const { lastResult } = await chrome.storage.local.get("lastResult");
  
    if (!lastResult) return;
  
    if (lastResult.error) {
      el.textContent = `Error: ${lastResult.error}`;
      return;
    }
  
    const pct = Math.round((lastResult.final_confidence || 0) * 100);
    el.innerHTML = `
      <div>AI-generated likelihood:</div>
      <div class="big">${pct}%</div>
      <div class="muted">Signals:</div>
      ${(lastResult.signals || []).map(s => `
        <div class="sig">
          <div><b>${s.name}</b> (${Math.round(s.score * 100)}%)</div>
          <div class="muted">${s.detail}</div>
        </div>
      `).join("")}
    `;
  }
  main();