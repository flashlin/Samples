---
skill: httpClient-csharp
description: HttpClient 使用規範：不使用 try-catch，讓例外向上傳播
tags: [csharp, HttpClient]
---

## HttpClient 使用規範

### 錯誤處理
- 呼叫 HttpClient 時，不要使用 try-catch 區塊包裹
- 讓例外自然向上傳播，由呼叫端決定如何處理

