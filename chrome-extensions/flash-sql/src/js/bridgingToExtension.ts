import { CHANGE_EVENT_NAME, DIV_TO_EXTENSION_EVENTS } from "@/models/typings";

type ResolveFunc = (data?: any) => void;

function appendSubDivEvents(
  eventName: string,
  eventData: any,
  resolve: ResolveFunc
) {
  const element = document.createElement("div");
  element.setAttribute("eventName", eventName);
  if (eventData != null) {
    element.setAttribute("eventData", JSON.stringify(eventData));
  }
  const tmpElement = document.getElementById(DIV_TO_EXTENSION_EVENTS);
  tmpElement!.appendChild(element);

  const mo = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mo.disconnect();
      const eventResult = element.getAttribute("eventResult");
      if (eventResult == null) {
        resolve();
        return;
      }
      resolve(JSON.parse(eventResult));
    });
  });
  mo.observe(element, {
    attributes: true,
    attributeFilter: ["eventResult"],
  });
}

export function dispatchEventToExtensionAsync(
  eventName: string,
  eventData?: any
): Promise<any> {
  return new Promise((resolve, reject) => {
    appendSubDivEvents(eventName, eventData, resolve);
    const event = new Event(CHANGE_EVENT_NAME, {
      bubbles: false,
      cancelable: false,
    });
    const element = document.getElementById(DIV_TO_EXTENSION_EVENTS);
    element!.dispatchEvent(event);
  });
}
