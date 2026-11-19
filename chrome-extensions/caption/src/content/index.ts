import { isTabBroadcast } from "../shared/message-guards";
import { PageCaptionController } from "./page-caption-controller";

const URL_POLL_MILLISECONDS = 1000;

let currentUrl = location.href;
let controller = new PageCaptionController(currentUrl);
controller.start();

async function handleNavigation(): Promise<void> {
  if (location.href === currentUrl) {
    return;
  }
  currentUrl = location.href;
  const previous = controller;
  controller = new PageCaptionController(currentUrl);
  await previous.dispose();
  controller.start();
}

setInterval(() => void handleNavigation(), URL_POLL_MILLISECONDS);
window.addEventListener("yt-navigate-finish", () => void handleNavigation());
window.addEventListener("popstate", () => void handleNavigation());

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id || !isTabBroadcast(message)) {
    return false;
  }
  void controller.handleBroadcast(message).finally(() => sendResponse(null));
  return true;
});
