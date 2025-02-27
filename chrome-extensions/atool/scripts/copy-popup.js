import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const distDir = path.join(rootDir, 'dist');

// 確保 dist 目錄存在
if (!fs.existsSync(distDir)) {
  console.error('dist 目錄不存在，請先執行 vite build');
  process.exit(1);
}

// 直接創建 popup.html
try {
  const popupPath = path.join(distDir, 'popup.html');
  
  // 獲取構建後的 CSS 和 JS 文件名
  const files = fs.readdirSync(path.join(distDir, 'assets'));
  const cssFile = files.find(file => file.endsWith('.css'));
  const jsFile = files.find(file => file.endsWith('.js') && !file.includes('content') && !file.includes('background'));
  
  if (!cssFile || !jsFile) {
    console.error('找不到構建後的 CSS 或 JS 文件');
    process.exit(1);
  }
  
  // 創建 popup.html 內容
  const popupContent = `<!doctype html>
<html lang="zh-TW">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="./vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Chrome Extension</title>
    <script type="module" crossorigin src="./assets/${jsFile}"></script>
    <link rel="stylesheet" crossorigin href="./assets/${cssFile}">
    <style>
      html, body {
        width: 800px;
        height: 600px;
        margin: 0;
        padding: 0;
        overflow: auto;
      }
      #app {
        width: 100%;
        height: 100%;
        box-sizing: border-box;
        padding: 16px;
      }
    </style>
  </head>
  <body>
    <div id="app"></div>
    <script>
      // 動態調整彈出視窗大小
      document.addEventListener('DOMContentLoaded', function() {
        // 獲取螢幕尺寸
        const screenWidth = window.screen.availWidth;
        const screenHeight = window.screen.availHeight;
        
        // 設定為螢幕的 90%
        const popupWidth = Math.floor(screenWidth * 0.9);
        const popupHeight = Math.floor(screenHeight * 0.9);
        
        // 調整視窗大小
        document.documentElement.style.width = \`\${popupWidth}px\`;
        document.documentElement.style.height = \`\${popupHeight}px\`;
        document.body.style.width = \`\${popupWidth}px\`;
        document.body.style.height = \`\${popupHeight}px\`;
      });
    </script>
  </body>
</html>`;
  
  // 寫入 popup.html
  fs.writeFileSync(popupPath, popupContent);
  console.log('成功創建 popup.html');
} catch (error) {
  console.error('創建 popup.html 時出錯:', error);
  process.exit(1);
}

// 確保 js 目錄存在
const jsDir = path.join(distDir, 'js');
if (!fs.existsSync(jsDir)) {
  fs.mkdirSync(jsDir, { recursive: true });
}

// 複製 content.js 和 background.js 到正確位置
try {
  const contentSrcPath = path.join(distDir, 'assets/content.js');
  const backgroundSrcPath = path.join(distDir, 'assets/background.js');
  const contentDestPath = path.join(distDir, 'js/content.js');
  const backgroundDestPath = path.join(distDir, 'js/background.js');
  
  // 如果構建後的文件存在，則複製到正確位置
  if (fs.existsSync(contentSrcPath)) {
    fs.copyFileSync(contentSrcPath, contentDestPath);
    console.log('已複製 content.js 到 js 目錄');
  } else {
    console.warn('找不到構建後的 content.js，創建空文件');
    fs.writeFileSync(contentDestPath, '// 內容腳本\nconsole.log("Chrome 擴充功能內容腳本已載入");');
  }
  
  if (fs.existsSync(backgroundSrcPath)) {
    fs.copyFileSync(backgroundSrcPath, backgroundDestPath);
    console.log('已複製 background.js 到 js 目錄');
  } else {
    console.warn('找不到構建後的 background.js，創建空文件');
    fs.writeFileSync(backgroundDestPath, '// 背景腳本\nconsole.log("Chrome 擴充功能背景腳本已啟動");');
  }
} catch (error) {
  console.error('處理腳本文件時出錯:', error);
}

// 確保 manifest.json 存在於 dist 目錄
try {
  const srcManifestPath = path.join(rootDir, 'public/manifest.json');
  const destManifestPath = path.join(distDir, 'manifest.json');
  
  if (fs.existsSync(srcManifestPath)) {
    // 讀取 manifest.json
    const manifest = JSON.parse(fs.readFileSync(srcManifestPath, 'utf8'));
    
    // 確保 action.default_popup 設置為 popup.html
    if (manifest.action) {
      manifest.action.default_popup = 'popup.html';
    }
    
    // 寫入 manifest.json 到 dist 目錄
    fs.writeFileSync(destManifestPath, JSON.stringify(manifest, null, 2));
    console.log('已複製並更新 manifest.json');
  } else {
    console.error('找不到 manifest.json 在 public 目錄中');
  }
} catch (error) {
  console.error('處理 manifest.json 時出錯:', error);
}

// 確保 images 目錄存在
const imagesDir = path.join(distDir, 'images');
if (!fs.existsSync(imagesDir)) {
  fs.mkdirSync(imagesDir, { recursive: true });
  console.log('已創建 images 目錄');
  
  // 複製圖標文件或創建臨時圖標
  const srcImagesDir = path.join(rootDir, 'public/images');
  if (fs.existsSync(srcImagesDir)) {
    // 複製所有圖標文件
    const iconFiles = fs.readdirSync(srcImagesDir);
    for (const file of iconFiles) {
      fs.copyFileSync(
        path.join(srcImagesDir, file),
        path.join(imagesDir, file)
      );
    }
    console.log('已複製圖標文件');
  } else {
    // 創建簡單的圖標文件
    const sizes = [16, 48, 128];
    for (const size of sizes) {
      fs.writeFileSync(
        path.join(imagesDir, `icon${size}.png`),
        Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeAJ5jHI2/wAAAABJRU5ErkJggg==', 'base64')
      );
    }
    console.log('已創建臨時圖標文件');
  }
}

console.log('構建後處理完成'); 