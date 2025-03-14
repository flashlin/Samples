namespace VimSharpLib;
using System.Text;

/// <summary>
/// VimSharp 類，用於管理多個 VimEditor 實例
/// </summary>
public class VimSharp
{
    private List<VimEditor> _editors = new List<VimEditor>();
    private VimEditor? _currentEditor;
    private bool _isRunning = true;

    /// <summary>
    /// 添加一個編輯器實例
    /// </summary>
    /// <param name="editor">要添加的編輯器</param>
    public void AddEditor(VimEditor editor)
    {
        
        _editors.Add(editor);
        
        // 如果這是第一個添加的編輯器，則自動設置為當前編輯器
        if (_currentEditor == null)
        {
            _currentEditor = editor;
        }
        
    }

    /// <summary>
    /// 設置焦點到指定的編輯器
    /// </summary>
    /// <param name="editor">要設置焦點的編輯器</param>
    public void FocusEditor(VimEditor editor)
    {
        if (_editors.Contains(editor))
        {
            _currentEditor = editor;
        }
    }

    /// <summary>
    /// 運行 VimSharp，處理所有編輯器的渲染和輸入
    /// </summary>
    public void Run()
    {
        // 初始化所有編輯器
        foreach (var editor in _editors)
        {
            editor.Initialize();
        }
        
        // 獲取控制台大小（使用第一個編輯器的控制台）
        var consoleDevice = _editors.FirstOrDefault()?.GetConsoleDevice();
        if (consoleDevice == null)
        {
            return; // 如果沒有編輯器，則直接返回
        }
        
        int consoleWidth = consoleDevice.WindowWidth;
        int consoleHeight = consoleDevice.WindowHeight;
        
        // 創建一個共享的緩衝區
        ColoredChar[,] screenBuffer = new ColoredChar[consoleHeight, consoleWidth];
        
        // 不需要手動初始化緩衝區
        // 每個編輯器的 Render 方法只會修改其 ViewPort 區域內的緩衝區
        // 這允許多個編輯器無干擾地共享同一個屏幕緩衝區
        
        // 主循環
        while (_isRunning)
        {
            // 先渲染非當前焦點的編輯器
            // 每個編輯器只會修改其 ViewPort 區域內的緩衝區，不會影響其他區域
            foreach (var editor in _editors)
            {
                if(editor == _currentEditor){
                    continue;
                }
                editor.Render(screenBuffer);
            }

            // 最後渲染當前焦點編輯器，確保它顯示在最上層
            if (_currentEditor != null)
            {
                _currentEditor.Render(screenBuffer);
                
                // 將緩衝區內容一次性輸出到控制台
                RenderScreenBuffer(screenBuffer, consoleDevice);
                
                _currentEditor.WaitForInput();
                
                // 檢查當前編輯器是否還在運行
                if (!_currentEditor.IsRunning)
                {
                    // 如果當前編輯器停止運行，則切換到下一個編輯器
                    _currentEditor = _editors.Where(x => x.IsRunning).FirstOrDefault();
                    if (_currentEditor == null)
                    {
                        // 如果沒有其他編輯器，則停止運行
                        _isRunning = false;
                    }
                }
            }
            else
            {
                // 如果沒有當前編輯器，則停止運行
                _isRunning = false;
            }
        }
    }
    
    /// <summary>
    /// 將屏幕緩衝區內容渲染到控制台
    /// </summary>
    private void RenderScreenBuffer(ColoredChar[,] screenBuffer, IConsoleDevice consoleDevice)
    {
        // 隱藏游標
        consoleDevice.Write("\x1b[?25l");
        
        // 創建一個緩衝區用於收集所有輸出
        var outputBuffer = new StringBuilder();
        
        int height = screenBuffer.GetLength(0);
        int width = screenBuffer.GetLength(1);
        
        // 遍歷每一行
        for (int y = 0; y < height; y++)
        {
            // 設置光標位置到行的起始位置
            outputBuffer.Append($"\x1b[{y + 1};1H");
            
            // 遍歷當前行的每個字符
            for (int x = 0; x < width; x++)
            {
                ColoredChar? c = screenBuffer[y, x];
                
                // 如果字符為空，使用默認空白字符
                if (c == null || c.Value.Char == '\0')
                {
                    outputBuffer.Append(ColoredChar.Empty.ToAnsiString());
                }
                else
                {
                    outputBuffer.Append(c.Value.ToAnsiString());
                }
            }
        }
        
        // 如果當前編輯器存在，設置光標位置到編輯器的光標位置
        if (_currentEditor != null)
        {
            outputBuffer.Append($"\x1b[{_currentEditor.Context.CursorY + 1};{_currentEditor.Context.CursorX + 1}H");
        }
        
        // 顯示游標
        outputBuffer.Append("\x1b[?25h");
        
        // 一次性輸出所有內容到控制台
        consoleDevice.Write(outputBuffer.ToString());
    }
} 