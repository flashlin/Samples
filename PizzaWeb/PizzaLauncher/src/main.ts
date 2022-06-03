import { createApp } from 'vue';
import { createPinia } from 'pinia';
import layout from "./Layout.vue";
import route from "./Router";
import "./Main.scss";

// import "bootstrap/dist/css/bootstrap.min.css";
// import "bootstrap";
// import 'bootstrap-icons/font/bootstrap-icons.css';

import PrimeVue from 'primevue/config';
import ToastService from 'primevue/toastservice';
import ConfirmationService from 'primevue/confirmationservice';
import { useFocus } from './directives/focus';

function program() {
  const app = createApp(layout);
  useFocus(app);
  app.use(PrimeVue);
  app.use(ToastService);
  app.use(ConfirmationService);
  app.use(createPinia());
  app.use(route);
  app.mount('#app');
}

program();

