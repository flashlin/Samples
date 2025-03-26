import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// 獲取當前文件的目錄
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

// 源文件和目標目錄
const publicDir = path.join(rootDir, 'public');
const distDir = path.join(rootDir, 'dist');

// 複製文件函數
function copyFile(src, dest) {
  fs.copyFileSync(src, dest);
  console.log(`已複製: ${path.relative(rootDir, src)} -> ${path.relative(rootDir, dest)}`);
}

// 複製目錄函數
function copyDir(src, dest) {
  // 確保目標目錄存在
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  // 讀取源目錄中的所有文件和子目錄
  const entries = fs.readdirSync(src, { withFileTypes: true });

  // 遍歷並複製每個文件和子目錄
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      // 遞歸複製子目錄
      copyDir(srcPath, destPath);
    } else {
      // 複製文件
      copyFile(srcPath, destPath);
    }
  }
}

// 主函數
function main() {
  console.log('開始後構建處理...');

  // 複製 manifest.json 到 dist 目錄
  const manifestSrc = path.join(publicDir, 'manifest.json');
  const manifestDest = path.join(distDir, 'manifest.json');
  copyFile(manifestSrc, manifestDest);

  // 複製 welcome.html 到 dist 目錄
  const welcomeSrc = path.join(publicDir, 'welcome.html');
  const welcomeDest = path.join(distDir, 'welcome.html');
  copyFile(welcomeSrc, welcomeDest);

  // 複製 popup.html 到 dist 目錄
  const popupSrc = path.join(publicDir, 'popup.html');
  const popupDest = path.join(distDir, 'popup.html');
  copyFile(popupSrc, popupDest);

  // 複製 images 目錄到 dist 目錄
  const imagesSrc = path.join(publicDir, 'images');
  const imagesDest = path.join(distDir, 'images');
  copyDir(imagesSrc, imagesDest);

  console.log('後構建處理完成！');
}

// 執行主函數
main(); 