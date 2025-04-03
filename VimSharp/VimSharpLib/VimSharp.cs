using System.Text;

namespace VimSharpLib;

/// <summary>
/// VimSharp 類，用於管理多個 VimEditor 實例
/// </summary>
public class VimSharp
{
    private readonly IConsoleDevice _consoleDevice;
    private readonly List<VimEditor> _editors = new();
    private VimEditor? _currentEditor;
    private bool _isRunning = true;

    public VimSharp(IConsoleDevice consoleDevice)
    {
        _consoleDevice = consoleDevice;
    }

    /// <summary>
    /// 添加一個編輯器實例
    /// </summary>
    /// <param name="editor">要添加的編輯器</param>
    public void AddEditor(VimEditor editor)
    {
        editor.Console = _consoleDevice;
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
        var screenBuffer = ColoredCharScreen.CreateScreenBuffer(_consoleDevice);

        // 主循環
        while (_isRunning)
        {
            // 渲染所有編輯器
            foreach (var editor in _editors)
            {
                if( editor == _currentEditor){
                    continue;
                }
                editor.Render(screenBuffer);
            }

            // 處理當前編輯器的輸入
            if (_currentEditor != null)
            {
                _currentEditor.Render(screenBuffer);
                WriteToConsole(_currentEditor, screenBuffer);
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

    private void WriteToConsole(VimEditor vimEditor, ColoredCharScreen screenBuffer)
    {
        // 創建一個緩衝區用於收集所有輸出
        var outputBuffer = new StringBuilder();
        outputBuffer.Append($"\x1b[0;0H");
        // 隱藏游標 (符合 Rule 12)
        outputBuffer.Append("\x1b[?25l");

        vimEditor.RenderBufferToConsole(screenBuffer, outputBuffer);

        vimEditor.Mode.AfterRender(outputBuffer);

        // 一次性輸出所有內容到控制台
        _consoleDevice.Write(outputBuffer.ToString());
    }
} 