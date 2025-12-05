---
name: query-customer-info
description: "查詢sbo客戶資訊"
---

## 使用方式

使用 `scripts/` 目錄中的 TypeScript 檔案來查詢客戶資料。

### 執行查詢

```bash
ts-node scripts/query-customer-info.ts
```

### API 說明

`queryCustomerInfo()` 函式會向 SBO 客戶資訊系統發送 POST 請求。

**查詢參數：**
- action: 1
- txtUserName: tflash
- txtPassword: abc

**回傳值：**
- 成功：包含 status、data、message 的物件
- 失敗：包含 status: 'error'、data: null、message 的物件

### 前置需求

```bash
npm install -g ts-node
npm install axios
```