namespace VimSharpLib;

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

        var screenBuffer = _editors[0].CreateScreenBuffer();

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
                _currentEditor.RenderToConsole(screenBuffer);
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
} 