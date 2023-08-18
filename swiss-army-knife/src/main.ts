//import './assets/main.css';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import router from './router';

import ElementPlus from 'element-plus'
import 'element-plus/theme-chalk/dark/css-vars.css'

import './db';
import { parseTsql } from './sqlex/tsql';
//import { Tsql } from './antlr/TSQL';
//const tsql = new Tsql();
parseTsql('select id from customer');

import { appDiv } from "./initMonkey";
const app = createApp(App);
app.use(createPinia());
app.use(router);
app.use(ElementPlus, { size: 'small', zIndex: 9998 })
//app.mount('#app');
app.mount(appDiv);