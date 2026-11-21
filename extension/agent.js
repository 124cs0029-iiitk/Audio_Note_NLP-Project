// Runs on the Meeting Agent page (127.0.0.1:8000).
// Hands the messages from the extension to the page's own code.

chrome.runtime.onMessage.addListener(msg => {
  window.postMessage({ source: "meeting-agent-bridge", ...msg }, "*");
});