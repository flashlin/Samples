import { createApp } from 'vue';
import { createPinia } from 'pinia';
import layout from "./Layout";
import route from "./Router";

import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap";
import 'bootstrap-icons/font/bootstrap-icons.css';

import { ElTable } from 'element-plus';

function program() {
  const app = createApp(layout);
  app.component(ElTable.name, ElTable);
  app.use(createPinia());
  app.use(route);
  app.mount('#app');
}

program();

