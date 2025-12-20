import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import './style.css'
import App from './App.vue'
import T1WebComponents from 't1-web-components'

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(T1WebComponents)

app.mount('#app')
