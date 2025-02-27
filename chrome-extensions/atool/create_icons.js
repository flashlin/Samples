const fs = require('fs');
const path = require('path');

// 確保目錄存在
const publicImagesDir = path.join(__dirname, 'public', 'images');
if (!fs.existsSync(publicImagesDir)) {
  fs.mkdirSync(publicImagesDir, { recursive: true });
}

// 這是一個預先生成的 16x16 像素的 PNG 圖像，顯示黃色 F 字母，白色背景
// 使用 Base64 編碼
const icon16Base64 = `
iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAOxAAADsQBlSsOGwAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAADDSURBVDiNpZIxDoJAEEXfLBYWxsLEwsrEztLGxMrGG3gEj+ARPIJHsLGx9AYewcbKxMKEwsLCYhcTCURZwEmm2Zn/Xv7MwL9Nk0iqS9pIepW0lnQys1FZfVBSHMdTYAGMgRZwB1bAzMx2hYAoikZABnSBFDgDJ+AKtIEHMDSzY+6fJMkAWAJd4AXMzWz/BpjZRdJe0tHMprkwTdMuMAdGQAYszezpA+SWAxtg4pxbFYVpmg6BBdAHrsA0y7Kz7/sVfWUvJJhQZz7AxWoAAAAASUVORK5CYII=
`;

// 將 Base64 字符串轉換為 Buffer
const imageData = icon16Base64.trim();
const imageBuffer = Buffer.from(imageData, 'base64');

// 將圖像數據寫入文件
fs.writeFileSync(path.join(publicImagesDir, 'icon16.png'), imageBuffer);

// 複製相同的圖像作為其他尺寸的圖標（在實際應用中，您應該為每個尺寸創建適當的圖像）
fs.writeFileSync(path.join(publicImagesDir, 'icon48.png'), imageBuffer);
fs.writeFileSync(path.join(publicImagesDir, 'icon128.png'), imageBuffer);

console.log('圖標文件已成功生成並保存到 public/images 目錄！'); 