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

// 複製 index.html 到 popup.html
try {
  const indexPath = path.join(distDir, 'index.html');
  const popupPath = path.join(distDir, 'popup.html');
  
  if (fs.existsSync(indexPath)) {
    fs.copyFileSync(indexPath, popupPath);
    console.log('成功將 index.html 複製到 popup.html');
  } else {
    console.error('index.html 不存在於 dist 目錄中');
    process.exit(1);
  }
} catch (error) {
  console.error('複製文件時出錯:', error);
  process.exit(1);
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

console.log('構建後處理完成'); 