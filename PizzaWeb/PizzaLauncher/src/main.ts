import { createApp } from 'vue';
import { createPinia } from 'pinia';
import layout from "./Layout";
import route from "./Router";

import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap";
import 'bootstrap-icons/font/bootstrap-icons.css';

function program() {
  const app = createApp(layout);
  app.use(createPinia());
  app.use(route);
  app.mount('#app');
}

program();

