---
marp: true
theme: uncover
class: invert
---
### Introducing AI into Company Operations

![w:500 h:300](ai.jpg)


---
# Goal

* 透過 AI 驅動的幫助提高内部 Domain 搜尋和傳承
* 提高我們的生產力
- 產生創意的內容 
- 提供 Code 輔助就像 Copilot
- 提供資料 Report 

---
<style scoped>
table {
  font-size: 24px;
}
</style>
# 解決方案

|建置方式 |廠商 |Description
|--|--|--
|綜合型雲端 |OpenAI, Azure,... |公司將資料上傳至雲端, 並且有提供封閉模型, 和提供訓練微調服務
|GPU 服務型雲端 |CoreWeave、Lambda、RunPod 和 CodLab |公司將資料上傳至雲端, 得自行準備模型, 自行訓練微調
|GPU 私有雲 |公司內部自行準備GPU 伺服器 |在內部自行訓練微調


---
<style scoped>
table {
  font-size: 24px;
}
</style>
# 硬體選項

|建置方式 |優點 |缺點
|--|--|--
|RTX 3090 24GB |便宜一點, 一張卡5萬  |記憶體嚴重不足, 需準備更多顯示卡, 電源消耗更多, 一張350W
|RTX A6000 48GB |記憶體大, 推斷算力也能服務給多 user, 比較省電, 300W | 價格比較貴, 一張卡18萬
|RTX A100 80GB |記憶體最大, 可服務給user 和開發, 300W  |價格45萬一張
|Mac Studio M2 Ultra 192GB |單一台記憶體最大 |問題資源比較少. 開源模型都是以linux 為主, 不一定有支援Mac硬體加速. 額外服務給 user 使用會比較慢


---

### 選 A6000 的原因
- nvidia platform 資源生態
遇到問題時候, 利用Google 進行搜尋可以獲得更豐富的資源和信息.
- 目前大語言模型所需要的記憶體越來越大.
(目前已知的最大模型是 320GB, 這還是壓縮過後的執行大小)
- 歷史上跟大有關的東西會朝小型化演進. 
- 考量大小平衡點跟服務 user.

---
### 硬體
- CPU: i7-12700H 20Core
- Memory: 32GB
- Disk: SSD 1.5TB
- GPU: A6000 48GB 
- OS: Ubuntu 22.04


---
## milestone1
- 建立一個問答型的 WebSite 第一版 (不具有私有 domain knowledge)
- 每當使用者輸入一個問題，系統將記錄下來，並提供回答。使用者必須進行投票（有幫助，無意見，沒幫助）
- 當使用者按下[沒幫助]後, 將提供一個建議答案的輸入框, 讓使用者輸入, 以供系統後續未來進行訓練或微調
- 當使用者按下[有幫助/無意見/沒幫助]按鈕之後, 才能進行下一輪的問題

---
## milestone2
- 收集現有的 domain knowledge 資料, 並將其整理為正確的內容
- 進行升級問答型的 WebSite 第二版 (具有私有 domain knowledge)

---
## milestone3
1. 建立 Code 專家型 WebSite 第一版, 專門協助回答 Code 的問題

---
## milestone4
- 建立資料問答型的 WebSite 
與 .csv 資料檔案連接, 提供跟資料內容有關的問答

---
## mileston5
- 將 Code AI 外掛上去到 VSCode, 就像 Copilot 輔助
- 製作 PPT 說明, 介紹給 All PD 並說明安裝步驟


---
## milestone6
- 將 domain knowledge 資料轉換為知識訓練資料的格式
- 轉換輸入的回饋答案為知識訓練資料的格式
- 進行再一次的模型訓練或微調
- 升級 WebSite 

---
## milestone7
1. 收集已輸入的問題和 Code 問題回饋, 以檢查是否需要進行進一步的訓練或微調
2. 等待外面開放訓練模型版本更新, 再一次進行升級迭代

---
# 成功的定義
根據現在收集的[有幫助/無意見/沒幫助]的次數來評估成果
* 如果[有幫助]的次數比[沒幫助]的多
* Chat + Data + Code = 每日查詢次數 100 次
我們可以視為成功

---
經過這次建置過程, 我們將會獲得下面的東西
- 收集到正確的 domain QA 資料
有了這些正確的資料, 下一代的 AI 就會更正確, (正確的資料才能產生正確的知識, 而不是甚麼都不整理, 一股腦丟給 AI)
- 知道 user 會問了哪些問題, 協助我們創造, 腦力激發,
- 至少有一台私有的 AI 可以問 Domain 問題而不必擔心機密外洩

