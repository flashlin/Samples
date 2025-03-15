# VimSharp 編輯器

VimSharp 是一個使用 C# 實現的簡易 Vim 風格編輯器，可在控制台環境中運行。

## 功能

- 基本的 Vim 模式：普通模式、插入模式、可視模式
- 文本編輯和游標移動
- 狀態列顯示當前模式
- 行號顯示
- 文件載入功能

## 使用方法

### 運行編輯器

```bash
# 直接啟動編輯器
dotnet run

# 帶文件參數啟動編輯器
dotnet run filename.txt
```

### 快捷鍵

#### 普通模式 (Normal Mode)

- `h`: 向左移動游標
- `j`: 向下移動游標
- `k`: 向上移動游標
- `l`: 向右移動游標
- `i`: 進入插入模式
- `v`: 進入可視模式
- `$`: 移動到行尾 (使用 Shift+4)

#### 插入模式 (Insert Mode)

- `Esc`: 返回普通模式
- 方向鍵: 移動游標
- `Backspace`: 刪除字符
- `Enter`: 插入新行

#### 可視模式 (Visual Mode)

- `Esc`: 返回普通模式
- `h/j/k/l`: 移動游標以選擇文本
- `y`: 複製選中內容並返回普通模式

## 工程結構

- `VimEditor.cs`: 核心編輯器類，管理文本內容和視窗渲染
- `ConsoleText.cs`: 行文本類
- `ColoredChar.cs`: 帶顏色的字符類
- `ViewArea.cs`: 視窗區域類
- `IConsoleDevice.cs`: 控制台設備介面
- `ConsoleDeviceAdapter.cs`: 控制台設備適配器實現
- `IVimMode.cs`: Vim 模式介面
- `VimNormalMode.cs`: 普通模式實現
- `VimInsertMode.cs`: 插入模式實現
- `VimVisualMode.cs`: 可視模式實現
- `IKeyPattern.cs`: 按鍵模式介面
- `Program.cs`: 主程序入口

## 架構特點

- 使用 Host.CreateApplicationBuilder 啟動
- 依賴注入模式
- 模式切換採用組合模式
- 視窗緩衝區渲染機制 