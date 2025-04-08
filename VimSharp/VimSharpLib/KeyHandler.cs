namespace VimSharpLib;

public class KeyHandler : IKeyHandler
{
    private readonly IConsoleDevice _consoleDevice;
    private Dictionary<IKeyPattern, Action<List<ConsoleKeyInfo>>> _defaultKeyPatterns = new();
    private readonly List<ConsoleKeyInfo> _keyBuffer = new();
    private readonly Dictionary<IKeyPattern, Action<IProgress>> _userKeyPressActions = new();
    private VimEditor? _editor;
    
    public KeyHandler(IConsoleDevice consoleDevice)
    {
        _consoleDevice = consoleDevice;
    }

    public void SetEditor(VimEditor editor)
    {
        _editor = editor;
    }

    public void InitializeKeyHandlers(Dictionary<IKeyPattern, Action<List<ConsoleKeyInfo>>> keyPatterns)
    {
        _defaultKeyPatterns = keyPatterns;
    }
    
    public string GetKeyBufferString()
    {
        return string.Join("", _keyBuffer.Select(k => k.KeyChar).Where(c => c != '\0'));
    }

    public void Clear()
    {
        _keyBuffer.Clear();
    }

    /// <summary>
    /// 添加按鍵處理動作
    /// </summary>
    /// <param name="keyPattern">按鍵模式</param>
    /// <param name="action">要執行的動作</param>
    public void AddOnKeyPress(IKeyPattern keyPattern, Action<IProgress> action)
    {
        _userKeyPressActions[keyPattern] = action;
    }
    
    /// <summary>
    /// 處理自定義按鍵
    /// </summary>
    /// <param name="keyInfo">按鍵信息</param>
    /// <returns>如果按鍵被處理則返回 true，否則返回 false</returns>
    public bool HandleUserKeyPress(ConsoleKeyInfo keyInfo)
    {
        if (_editor == null)
        {
            return false;
        }
        var tempBuffer = new List<ConsoleKeyInfo> { keyInfo };
        foreach (var keyPattern in _userKeyPressActions.Keys)
        {
            if (keyPattern.IsMatch(tempBuffer))
            {
                var progress = new ProgressReporter(_editor);
                _userKeyPressActions[keyPattern].Invoke(progress);
                _keyBuffer.Clear();
                return true;
            }
        }
        return false;
    }

    public void PressKey(ConsoleKeyInfo keyInfo)
    {
        // 首先檢查是否有自定義按鍵處理
        if (HandleUserKeyPress(keyInfo))
        {
            return;
        }
        
        _keyBuffer.Add(keyInfo);
        HandleInputKey();
    }

    public void WaitForInput()
    {
        var keyInfo = _consoleDevice.ReadKey(intercept: false);
        
        // 首先檢查是否有自定義按鍵處理
        if (HandleUserKeyPress(keyInfo))
        {
            return;
        }
        
        _keyBuffer.Add(keyInfo);
        HandleInputKey();
    }

    private void HandleInputKey()
    {
        // 計算匹配的模式數量
        var matchCount = 0;
        IKeyPattern? matchedPattern = null;
        List<IKeyPattern> matchedPatterns = new();
        
        foreach (var pattern in _defaultKeyPatterns.Keys)
        {
            if (pattern.IsMatch(_keyBuffer))
            {
                matchCount++;
                matchedPattern = pattern;
                matchedPatterns.Add(pattern);
            }
        }

        if (matchCount == 2 && matchedPattern is AnyKeyPattern)
        {
            var nonAnyKeyPattern = matchedPatterns.First(x => x is not AnyKeyPattern);
            _defaultKeyPatterns[nonAnyKeyPattern].Invoke(_keyBuffer.ToList());
            _keyBuffer.Clear();
            return;
        }
        
        // 如果只有一個模式匹配，執行對應的操作
        if (matchCount == 1 && matchedPattern != null)
        {
            _defaultKeyPatterns[matchedPattern].Invoke(_keyBuffer.ToList());
            _keyBuffer.Clear();
        }
        else if (matchCount == 0 && _keyBuffer.Count >= 3)
        {
            _keyBuffer.Clear();
        }
    }
}