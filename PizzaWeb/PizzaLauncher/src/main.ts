import { createApp } from 'vue';
import { createPinia } from 'pinia';
import layout from "./Layout";
import route from "./Router";
import PrimeVue from 'primevue/config';

function program() {
  const app = createApp(layout);
  app.use(createPinia());
  app.use(route);
  app.use(PrimeVue);
  app.mount('#app');
}

program();

