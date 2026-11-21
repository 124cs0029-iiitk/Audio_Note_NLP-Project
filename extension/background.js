// Passes messages from the Google Meet page to the Meeting Agent page.

chrome.runtime.onMessage.addListener((msg, sender) => {
  if (sender.tab && sender.tab.url && sender.tab.url.startsWith("https://meet.google.com/")) {
    relay(msg);
  }
});

async function relay(msg) {
  const tabs = await chrome.tabs.query({
    url: ["http://127.0.0.1:8000/*", "http://localhost:8000/*"],
  });
  for (const tab of tabs) {
    chrome.tabs.sendMessage(tab.id, msg).catch(() => {});   // ignore tabs that are not ready
  }
}