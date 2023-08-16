import './assets/main.css';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import router from './router';

import './db';
import { parseTsql } from './sqlex/tsql';
//import { Tsql } from './antlr/TSQL';
//const tsql = new Tsql();
parseTsql('select id from customer');


// const app = createApp(App);
// app.use(createPinia());
// app.use(router);

// app.mount('#app');

import "./init"