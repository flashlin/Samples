import { App } from "vue";

export function useFocus(app: App<Element>) {
  app.directive("focus", {
    mounted(el: HTMLElement) {
      el.focus();
    },
  });
}
