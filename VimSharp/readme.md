# VimSharp - 簡易 Vim 編輯器實現

這是一個簡單的 Vim 風格編輯器的 C# 實現，提供基本的文字編輯功能和模式切換。

## 專案結構

```
VimSharpSolution.sln                  # 解決方案檔案
│
├── VimSharp/                         # 主程式專案
│   ├── Program.cs                    # 程式進入點，初始化並啟動編輯器
│   └── VimSharp.csproj               # 專案設定檔
│
└── VimSharpLib/                      # 核心庫專案
    ├── ConsoleCharacter.cs           # 控制台字符類，表示單個字符及其顯示屬性
    ├── ConsoleContext.cs             # 編輯器上下文，管理文字內容和光標位置
    ├── ConsoleRender.cs              # 渲染器，負責將文字渲染到控制台
    ├── ConsoleText.cs                # 文字行類，管理單行文字的字符集合
    ├── RenderArgs.cs                 # 渲染參數類，包含渲染所需的位置和文字信息
    ├── VimEditEditor.cs              # 編輯模式實現，處理文字輸入和編輯操作
    ├── VimEditor.cs                  # 主編輯器類，管理模式切換和整體流程
    └── VimSharpLib.csproj            # 庫專案設定檔
```

## 檔案說明

### VimSharp 專案

- **Program.cs**: 程式的入口點，初始化 VimEditor 並啟動編輯器。

### VimSharpLib 專案

- **ConsoleCharacter.cs**: 定義單個字符及其顯示屬性（顏色、背景色）。
- **ConsoleContext.cs**: 管理編輯器的上下文，包括光標位置和文字內容。
- **ConsoleRender.cs**: 負責將文字渲染到控制台，處理顏色和位置。
- **ConsoleText.cs**: 管理單行文字的字符集合，提供文字操作方法。
- **RenderArgs.cs**: 定義渲染所需的參數，包括位置和文字信息。
- **VimEditEditor.cs**: 實現編輯模式，處理用戶輸入、文字編輯和光標移動。
- **VimEditor.cs**: 主編輯器類，管理模式切換和整體流程控制。

## 功能說明

- **普通模式**: 按 `I` 鍵進入編輯模式，按 `Q` 鍵退出編輯器。
- **編輯模式**: 
  - 支持文字輸入
  - 支持退格鍵刪除文字
  - 支持方向鍵移動光標
  - 按 `Esc` 鍵返回普通模式
  - 支持 Big5 編碼中文字符處理，中文字符佔用兩個位置

## 編輯程式碼注意事項

1. **文字處理**:
   - 所有文字操作都通過 `ConsoleText` 和 `ConsoleCharacter` 類進行
   - 文字修改後需要重新渲染以顯示變化

2. **光標管理**:
   - 光標位置由 `ConsoleContext` 的 `X` 和 `Y` 屬性管理
   - 移動光標時需確保不超出文字範圍

3. **渲染機制**:
   - 每次文字變更後需調用 `_render.Render` 方法更新顯示
   - 渲染後需使用 `Console.SetCursorPosition` 設置光標位置

4. **模式切換**:
   - 在 `VimEditor` 中按 `I` 進入編輯模式
   - 在 `VimEditEditor` 中按 `Esc` 返回普通模式

5. **多行編輯**:
   - 確保在訪問 `Texts` 集合前檢查索引是否有效
   - 上下移動光標時需調整 `X` 位置以適應不同行的長度

6. **中文字符處理**:
   - 使用 Big5 編碼處理中文字符
   - 中文字符在顯示時佔用兩個位置
   - 需要區分顯示位置和實際字符索引
   - 光標移動時需考慮中文字符的寬度
   - 提供了輔助方法計算字符寬度和顯示位置

7. **擴展建議**:
   - 添加更多 Vim 命令支持
   - 實現文件保存和加載功能
   - 添加語法高亮支持
   - 實現搜索和替換功能
