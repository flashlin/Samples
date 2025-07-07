// getTxtFiles.ts
// pnpm install axios dotenv
// pnpm install -D @types/axios @types/node
import axios from 'axios';
import * as dotenv from 'dotenv';

// 加載 .env 檔案中的環境變數
dotenv.config();

const GITLAB_HOST = 'https://git.coreop.net';
const PROJECT_PATH_ENCODED = encodeURIComponent('b2c/sqlboxer-cache'); // 專案路徑需要編碼
const BRANCH = 'master';
const DIRECTORY = ''; // 目錄名稱

const GITLAB_USER = process.env.GITLAB_USER;
const GITLAB_TOKEN = process.env.GITLAB_TOKEN;

async function getDatabaseNamesByTxtFileNames() {
    if (!GITLAB_USER || !GITLAB_TOKEN) {
        console.error('錯誤：請在 .env 檔案中設定 GITLAB_USER 和 GITLAB_TOKEN。');
        return;
    }

    // GitLab API 取得儲存庫樹狀結構的端點
    // path 參數用於指定要查詢的目錄
    const url = `${GITLAB_HOST}/api/v4/projects/${PROJECT_PATH_ENCODED}/repository/tree?path=${DIRECTORY}&ref=${BRANCH}`;

    const response = await axios.get(url, {
        headers: {
            'Private-Token': GITLAB_TOKEN, // 使用私有存取令牌進行驗證
            // 'User-Agent': GITLAB_USER // 雖然不強制，但有些API服務會建議提供
        }
    });

    const files = response.data;
    console.log("files", files);

    if (files && Array.isArray(files)) {
        const txtFiles = files
            .filter((file: any) => file.type === 'blob' && file.name.endsWith('.txt'))
            .map((file: any) => file.name);

        const databaseNames = txtFiles.map((fileName: string) => {
            const databaseName = fileName.split('.')[0];
            return databaseName;
        });

        return databaseNames;
    } else {
        throw new Error('錯誤：GitLab API 返回的資料格式不正確。');
    }
}

async function main() {
    console.log('開始取得資料庫名稱...');
    const databaseNames = await getDatabaseNamesByTxtFileNames();
        
    console.log('找到的資料庫名稱:', databaseNames);
    
    // 確保 data 目錄存在
    const fs = await import('fs/promises');
    const path = await import('path');
    
    const dataDir = path.join(process.cwd(), 'data');
    try {
        await fs.mkdir(dataDir, { recursive: true });
    } catch (error) {
        // 目錄已存在，忽略錯誤
    }
    
    // 將資料庫名稱陣列序列化到 JSON 檔案
    const outputPath = path.join(dataDir, 'databaseNames.json');
    const jsonData = JSON.stringify(databaseNames, null, 2);
    await fs.writeFile(outputPath, jsonData, 'utf8');
    
    console.log(`資料庫名稱已成功寫入: ${outputPath}`);
}

// 執行主函數
main();
