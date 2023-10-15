//import './assets/main.css'
import "highlight.js/styles/dark.css";
import "@/assets/tailwind.css";
import "@icon-park/vue-next/styles/index.css";

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'


if (process.env.NODE_ENV === 'development') {
   import('@/mocks/index').then((module) => {
      const worker = module.default;
      worker.start();
   });
}

const app = createApp(App)

app.use(createPinia())
app.use(router)

app.mount('#app')
