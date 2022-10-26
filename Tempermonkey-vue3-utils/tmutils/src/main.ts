import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";
import router from "./router";

import "./assets/main.css";

import PrimeVue from 'primevue/config';
import "primevue/resources/themes/saga-blue/theme.css"; //theme
import "primevue/resources/primevue.min.css"; //core CSS
import "primeicons/primeicons.css"; //icons
import "primevue/resources/themes/bootstrap4-dark-purple/theme.css";

const app = createApp(App);
app.use(PrimeVue);
app.use(createPinia());
app.use(router);

//app.mount("#app");
const div = document.querySelector("#app");
if( div != null) {
   console.log("mount logo");
   app.mount(div);
} else {
   console.error("mount logo fail");
}
