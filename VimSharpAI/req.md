# C# Console 專案：VimEditor 實作規範

## 目標
建立一個 C# Console 專案，實作 `VimEditor`，並使用 `Host.CreateApplicationBuilder(args)` 啟動編輯器。

## 必要類別
專案應包含以下類別與介面：
- `VimEditor`
- `ConsoleText`
- `ColoredChar`
- `ViewArea`
- `VimNormalMode`
- `VimVisualMode`
- `VimInsertMode`
- `IConsoleDevice`
- `IVimMode`
- `IKeyPattern`

---

## `VimEditor` 類別
負責管理編輯器內容與游標位置。

### 屬性
- `List<ConsoleText> Texts`：儲存檔案內容，每行為一個 `ConsoleText`。
- `int CursorX` / `int CursorY`：游標目前在螢幕上的 X/Y 位置。
- `int OffsetX` / `int OffsetY`：內容的 X/Y 偏移量，決定 `Texts` 於 `ViewPort` 中的起始顯示位置。
- `ViewArea ViewPort`：限制 `Texts` 顯示範圍。
- `bool IsStatusBarVisible`：是否顯示狀態列。
- `ConsoleText StatusBar`：狀態列內容。
- `bool IsRelativeNumberVisible`：是否顯示相對行數。
- `IConsoleDevice Console`：控制台裝置。

### 方法
- `void OpenFile(string file)`：載入檔案內容至 `Texts`。
- `void Render(ColoredChar[,]? screenBuffer=null)`
  - 若 `screenBuffer` 為 `null`，則依 `Console.WindowWidth` 和 `Console.WindowHeight` 建立緩衝區，初始化為 `ColoredChar.Empty`。
  - 依 `ViewPort` 位置輸出 `Texts`。
  - 若 `IsStatusBarVisible`，則輸出 `StatusBar` 至 `ViewPort` 底部。
  - 若 `IsRelativeNumberVisible`，則顯示相對行數區域。
- `void WriteToConsole(ColoredChar[,]? screenBuffer)`
  - 將 `screenBuffer` 內容完整輸出至螢幕，無需額外邏輯。
- 游標移動方法：
  - `void MoveCursorLeft()`
  - `void MoveCursorRight()`
  - `void MoveCursorUp()`
  - `void MoveCursorDown()`
  - 移動前需檢查 `ViewPort` 範圍，超出則調整 `OffsetX` 或 `OffsetY`。

---

## `ViewArea` 類別
- 規範內容顯示範圍。
- 屬性：`X`, `Y`, `Width`, `Height`。

---

## `ConsoleText` 類別
- 屬性：
  - `ColoredChar[] Chars`：每行的字元陣列。
  - `int Width`：字元陣列長度 (`Chars.Length`)。

---

## `ColoredChar` 類別
- 屬性：
  - `char Char`：字元內容。
  - `ConsoleColor Foreground`：前景色。
  - `ConsoleColor Background`：背景色。
- 預設常數：
  - `static readonly ColoredChar ViewEmpty = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.DarkGray);`
  - `static readonly ColoredChar Empty = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.Black);`

---

## `IConsoleDevice` 介面
不可變更。

```csharp
public interface IConsoleDevice
{
    int WindowWidth { get; }
    int WindowHeight { get; }
    void SetCursorPosition(int left, int top);
    void Write(string value);
    ConsoleKeyInfo ReadKey(bool intercept);
}
```

---

## `IVimMode` 介面
`VimNormalMode`, `VimVisualMode`, `VimInsertMode` 必須實作此介面。

```csharp
public interface IVimMode
{
    void WaitForInput();
    VimEditor Instance { get; set; }
}
```

模式切換時，應使用以下方式：
```csharp
Instance.Mode = new VimVisualMode { Instance = Instance };
```

---

## `IKeyPattern` 介面
```csharp
public interface IKeyPattern
{
    bool IsMatch(List<ConsoleKey> keyBuffer);
}
```

---

## `VimNormalMode::WaitForInput()`
核心邏輯不可更改。

```csharp
public void WaitForInput()
{
    var keyInfo = Instance.Console.ReadKey(intercept: true);
    _keyBuffer.Add(keyInfo.Key);
    
    int matchCount = 0;
    IKeyPattern? matchedPattern = null;
    foreach (var pattern in _keyPatterns.Keys)
    {
        if (pattern.IsMatch(_keyBuffer))
        {
            matchCount++;
            matchedPattern = pattern;
        }
    }
    
    if (matchCount == 1 && matchedPattern != null)
    {
        _keyPatterns[matchedPattern].Invoke();
        _keyBuffer.Clear();
    }
    else if (matchCount == 0 && _keyBuffer.Count >= 3)
    {
        _keyBuffer.Clear();
    }
}
```

---

## 其他核心功能
- 取得目前游標的文本座標：
  ```csharp
  int actualX = Instance.GetActualTextX();
  int actualY = Instance.GetActualTextY();
  ```
- 取得目前游標所在行：
  ```csharp
  var currentLine = Instance.Texts[Instance.GetActualTextY()];
  ```
- 游標向右移動：
  ```csharp
  Instance.MoveCursorRight();
  ```

### `MoveCursorToEndOfLine()` 優化
```csharp
public void MoveCursorToEndOfLine()
{
    Instance.CursorX = Instance.Texts[Instance.GetActualTextY()].Width - 1;
}
```

---

## 簡潔程式碼規範
- **避免重複程式碼**，利用共用函式來簡化邏輯。
- **保持程式結構清晰**，確保可讀性。
- **遵循單一職責原則**，讓類別與方法各司其職。

---

這樣的規範讓 `VimEditor` 更易於擴展與維護，確保符合 Vim 模式切換、游標移動與內容顯示的核心需求。

