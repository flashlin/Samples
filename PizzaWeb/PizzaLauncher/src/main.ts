import { createApp } from 'vue';
import { createPinia } from 'pinia';
import layout from "./Layout";
import route from "./Router";
import "./Main.scss";

// import "bootstrap/dist/css/bootstrap.min.css";
// import "bootstrap";
// import 'bootstrap-icons/font/bootstrap-icons.css';

//import { elComponents, elPlugins } from "@/plugins/elementui";
import PrimeVue from 'primevue/config';

function program() {
  const app = createApp(layout);
  // elComponents.forEach(component => {
  //   app.component(component.name, component);
  // });
  // elPlugins.forEach(plugin => {
  //   app.use(plugin);
  // });
  app.use(PrimeVue);
  app.use(createPinia());
  app.use(route);
  app.mount('#app');
}

program();

