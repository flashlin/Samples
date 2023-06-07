const fs = require('fs');
const path = require('path');

function copyFolderSync(source, target) {
  if (!fs.existsSync(target)) {
    fs.mkdirSync(target);
  }

  const files = fs.readdirSync(source);
  files.forEach((file) => {
    const sourcePath = path.join(source, file);
    const targetPath = path.join(target, file);
    if (fs.statSync(sourcePath).isDirectory()) {
      copyFolderSync(sourcePath, targetPath);
    } else {
      console.log(`copy ${sourcePath} to ${targetPath}`)
      fs.copyFileSync(sourcePath, targetPath);
    }
  });
}

// 從命令行參數獲取來源資料夾和目標資料夾
const sourceFolder = process.argv[2];
const targetFolder = process.argv[3];

// 檢查參數是否存在
if (!sourceFolder || !targetFolder) {
  console.log('please provide source or target path');
  process.exit(1);
}

copyFolderSync(sourceFolder, targetFolder);
