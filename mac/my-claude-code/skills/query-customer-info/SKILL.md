---
name: query-customer-info
description: "查詢sbo客戶資訊"
---

在 command shell 中建立以下檔案
`
import axios from 'axios';
import { URLSearchParams } from 'url';

/**
 * 透過 POST 請求發送固定的表單數據至客戶資訊 API。
 * * 參數 (固定):
 * action=1
 * txtUserName={{user}}
 * txtPassword={{password}}
 * * @returns 包含 API 狀態和返回數據的物件。
 */
export async function queryCustomerInfo() {
    const url = 'http://leo-a01.sbotry.com:8088/default.aspx';
    
    // 1. 準備表單數據 (使用 URLSearchParams 確保正確編碼)
    const formData = new URLSearchParams();
    formData.append('action', '1');
    formData.append('txtUserName', 'tflash');
    formData.append('txtPassword', 'abc');

    try {
        // 2. 執行 POST 請求
        const response = await axios.post(url, formData.toString(), {
            headers: {
                // 設定正確的 Content-Type 以模擬表單提交
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        });

        // 3. 返回 API 響應的狀態碼和數據 (通常是 HTML 或 JSON)
        return {
            status: response.status,
            data: response.data,
            message: '客戶資訊查詢成功。'
        };

    } catch (error) {
        // 處理錯誤情況
        let errorMessage = '客戶資訊查詢失敗，請檢查網路或 API 狀態。';
        if (axios.isAxiosError(error) && error.response) {
            errorMessage += ` 伺服器狀態碼: ${error.response.status}.`;
        } else if (error instanceof Error) {
             errorMessage += ` 錯誤訊息: ${error.message}.`;
        }
        
        return {
            status: 'error',
            data: null,
            message: errorMessage
        };
    }
}
`

# 安裝 ts-node
npm install -g ts-node 

# 執行函式 (您需要一個入口點來呼叫 queryCustomerInfo)
ts-node query-customer.ts