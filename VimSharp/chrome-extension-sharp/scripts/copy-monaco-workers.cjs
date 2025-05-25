// 自動複製 Monaco Editor worker 檔案到 public/monaco (使用 min 目錄下的 UMD 版本)
const fs = require('fs')
const path = require('path')

const workers = [
  {
    src: 'node_modules/monaco-editor/min/vs/editor/editor.worker.js',
    dest: 'public/monaco/editor.worker.js',
  },
  {
    src: 'node_modules/monaco-editor/min/vs/language/json/json.worker.js',
    dest: 'public/monaco/json.worker.js',
  },
  {
    src: 'node_modules/monaco-editor/min/vs/language/css/css.worker.js',
    dest: 'public/monaco/css.worker.js',
  },
  {
    src: 'node_modules/monaco-editor/min/vs/language/html/html.worker.js',
    dest: 'public/monaco/html.worker.js',
  },
  {
    src: 'node_modules/monaco-editor/min/vs/language/typescript/ts.worker.js',
    dest: 'public/monaco/ts.worker.js',
  },
]

for (const { src, dest } of workers) {
  const destDir = path.dirname(dest)
  if (!fs.existsSync(destDir)) {
    fs.mkdirSync(destDir, { recursive: true })
  }
  fs.copyFileSync(src, dest)
  console.log(`Copied ${src} -> ${dest}`)
} 