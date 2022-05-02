import { createApp } from 'vue';
import { createPinia } from 'pinia';
import 'normalize.css/normalize.css';
import main from '@/views/main';

function program() {
  const app = createApp(main);
  app.use(createPinia());
  app.mount('#app');
}

program();
