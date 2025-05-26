import { createApp } from 'vue'
import './tailwind.css'
import './style.css'
import App from './App.vue'
import router from './router'
import EditorWorker from './monaco-workers/editor.worker?worker'
import JsonWorker from './monaco-workers/json.worker?worker'
import CssWorker from './monaco-workers/css.worker?worker'
import HtmlWorker from './monaco-workers/html.worker?worker'
import TsWorker from './monaco-workers/ts.worker?worker'

// Monaco Editor worker 設定 (ESM)
self.MonacoEnvironment = {
  getWorker: function (_moduleId, label) {
    switch (label) {
      case 'json':
        return new JsonWorker();
      case 'css':
      case 'scss':
      case 'less':
        return new CssWorker();
      case 'html':
      case 'handlebars':
      case 'razor':
        return new HtmlWorker();
      case 'typescript':
      case 'javascript':
        return new TsWorker();
      default:
        return new EditorWorker();
    }
  }
}

createApp(App).use(router).mount('#app')
