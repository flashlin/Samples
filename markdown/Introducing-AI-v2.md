---
marp: true
theme: uncover
class: invert
paginate: true
---
### Introducing AI into Company Operations

![w:500 h:300](ai.jpg)


---
##### Goal 
##### 透過 AI 驅動的幫助, 來提高我們的生產力 Boosting our productivity with the assistance of AI-driven solutions
- 幫助内部 Domain 知識的搜尋和傳承知識.
 Facilitating the search for internal domain knowledge and the transmission of knowledge
- 產生創意的內容
Generating creative content 
- ~~Private Code Copilot~~
- ~~Data Report~~
- ~~Translate into multiple languages~~

---
<style scoped>
table {
  font-size: 24px;
}
</style>
### Solution

|Type |Vendor |Description
|--|--|--
|綜合型雲端<br> Integrated Cloud |OpenAI, Azure,... |公司將資料上傳至雲端, 並且有提供封閉模型, 和提供訓練微調服務 <br> The company uploads data to the cloud, offers proprietary models, and provides training and fine-tuning services
|GPU 服務型雲端<br> GPU-as-a-Service Cloud |CoreWeave、Lambda、RunPod and CodLab |公司將資料上傳至雲端, 得自行準備模型, 自行訓練微調<br>The company uploads data to the cloud and is responsible for preparing and training the fine-tuning model on its own.
|GPU 私有雲<br> GPU Private Cloud|公司內部自行準備GPU 伺服器<br>The company prepares GPU servers |在內部自行訓練微調 <br> Internally prepare and fine-tune models.


---
<style scoped>
table {
  font-size: 24px;
}
</style>
### Hardware Options

|2x<br>RTX-4090 24GB|2x<br> RTX-A6000 48GB |2x<br>RTX-A100 80GB
|--|--|--
|$ for Mid-Size Models(7B, 13B) |More GPU Memory for Large Models(35B, 70B) |Max Performance with More Large Models (130B~)
|16-Cores CPU |32-Cores CPU|64-Cores CPU
|128 GB of system memory |256 GB of system memory |256 GB of system memory 

---
### 成功的定義 Definition of Success
根據現在收集的[有幫助/無意見/沒幫助]的次數來評估成果
Evaluate the outcome based on the number of [helpful/neutral/unhelpful] ratings collected at present.
* 如果[有幫助]的次數比[沒幫助]的多
If the number of [helpful] ratings is greater than [unhelpful]
* 每日查詢次數大於 100 次
daily query count exceeds 100 queries

我們可以視為成功 We can consider it a success.


---
### milestone1
- 建立一個問答型的WebSite (不具有私有 domain knowledge)
Create a question-and-answer-based website(without private domain knowledge)

---
- 每當使用者輸入一個問題，系統將記錄問題，並提供回答. 使用者必須進行投票（有幫助，無意見，沒幫助）, 才能繼續問下一個問題.
Whenever a user enters a question, the system will record the question and provide an answer. Users must vote (helpful, neutral, unhelpful) before they can ask the next question.

---
- 當使用者按下[沒幫助]後, 將提供一個輸入框, 讓使用者輸入建議答案, 以供系統後續未來進行訓練或微調
When a user selects [unhelpful], a text input box will be provided for the user to enter a suggested answer, which can be used for future training or fine-tuning of the system

---
## milestone2
- 收集現有的 domain knowledge 資料, 並將其整理為正確的內容
Collect existing domain knowledge data and organize it into accurate content.
- 進行升級問答型的 WebSite (具有私有 domain knowledge)
Upgrade the question-and-answer website with proprietary domain knowledge.

---
## milestone3
- 收集已輸入的問題回饋, 再檢查是否需要進行進一步的訓練或微調
Collect the entered questions and feedback, then check whether further training or fine-tuning is required.
- 等待外面開放訓練模型版本更新, 再一次進行升級迭代
Wait for the release of the external training model version and proceed with another upgrade iteration.

---
### Current Challenges:
- What types of domain knowledge data should we collect?
- How can we collect accurate domain knowledge data?
- Collect high-quality knowledge content or refine the content to make it of higher quality.

---
##### 經過這次建置過程, 我們將會獲得下面的東西
Through this construction process, we will obtain the following
- 收集到正確的 domain QA 資料

Accumulate accurate private domain QA data
有了這些正確的資料, 下一代的 AI 就會更正確
With this accurate data, the next generation of AI will be more precise

---
- 知道 user 會問了哪些問題, 協助我們創造, 腦力激發.
Understand what questions users are likely to ask, aiding our creativity and brainstorming

---
- 至少有一台私有的 AI 可以問 Domain 問題而不必擔心機密外洩
Have at least one private AI capable of answering private domain questions without concerns about confidentiality leaks.
