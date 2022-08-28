import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";

import "./assets/main.css";

const app = createApp(App);
app.use(createPinia());

//app.mount("#app");
const div = document.querySelector(".logo");
if( div != null) {
   console.log("mount logo");
   app.mount(div);
} else {
   console.error("mount logo fail");
}
