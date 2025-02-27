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

// 複製 index.html 到 popup.html 並修復路徑
try {
  const indexPath = path.join(distDir, 'index.html');
  const popupPath = path.join(distDir, 'popup.html');
  
  if (fs.existsSync(indexPath)) {
    // 讀取 index.html 內容
    let content = fs.readFileSync(indexPath, 'utf8');
    
    // 修復路徑，將 /assets/ 改為 ./assets/
    content = content.replace(/src="\/assets\//g, 'src="./assets/');
    content = content.replace(/href="\/assets\//g, 'href="./assets/');
    content = content.replace(/href="\/vite.svg"/g, 'href="./vite.svg"');
    
    // 寫入 popup.html
    fs.writeFileSync(popupPath, content);
    console.log('成功將 index.html 複製到 popup.html 並修復路徑');
  } else {
    console.error('index.html 不存在於 dist 目錄中');
    process.exit(1);
  }
} catch (error) {
  console.error('複製文件時出錯:', error);
  process.exit(1);
}

// 確保 js 目錄存在
const jsDir = path.join(distDir, 'js');
if (!fs.existsSync(jsDir)) {
  fs.mkdirSync(jsDir, { recursive: true });
}

// 複製 content.js 和 background.js 到正確位置
try {
  const contentSrcPath = path.join(distDir, 'js/content.js');
  const backgroundSrcPath = path.join(distDir, 'js/background.js');
  
  // 如果文件不存在，創建空文件
  if (!fs.existsSync(contentSrcPath)) {
    console.warn('找不到 content.js，創建空文件');
    fs.writeFileSync(contentSrcPath, '// 內容腳本\nconsole.log("Chrome 擴充功能內容腳本已載入");');
  }
  
  if (!fs.existsSync(backgroundSrcPath)) {
    console.warn('找不到 background.js，創建空文件');
    fs.writeFileSync(backgroundSrcPath, '// 背景腳本\nconsole.log("Chrome 擴充功能背景腳本已啟動");');
  }
  
  console.log('已確保 content.js 和 background.js 存在');
} catch (error) {
  console.error('處理腳本文件時出錯:', error);
}

// 確保 manifest.json 中的路徑正確
try {
  const manifestPath = path.join(distDir, 'manifest.json');
  
  if (fs.existsSync(manifestPath)) {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    
    // 確保 action.default_popup 設置為 popup.html
    if (manifest.action) {
      manifest.action.default_popup = 'popup.html';
    }
    
    // 寫回修改後的 manifest.json
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
    console.log('已更新 manifest.json');
  } else {
    console.warn('manifest.json 不存在於 dist 目錄中，請確保它已正確配置');
  }
} catch (error) {
  console.error('處理 manifest.json 時出錯:', error);
}

// 確保 images 目錄存在
const imagesDir = path.join(distDir, 'images');
if (!fs.existsSync(imagesDir)) {
  fs.mkdirSync(imagesDir, { recursive: true });
  console.log('已創建 images 目錄');
  
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

console.log('構建後處理完成'); 