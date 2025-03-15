namespace VimSharpLib;

public class KeyHandler
{
    private readonly IConsoleDevice _consoleDevice;
    private Dictionary<IKeyPattern, Action> _keyPatterns = new();
    private readonly List<ConsoleKey> _keyBuffer = new();
    
    public KeyHandler(IConsoleDevice consoleDevice)
    {
        _consoleDevice = consoleDevice;
    }

    public void InitializeKeyPatterns(Dictionary<IKeyPattern, Action> keyPatterns)
    {
        _keyPatterns = keyPatterns;
    }
    
    public string GetKeyBufferString()
    {
        return string.Join("", _keyBuffer.Select(k => k.ToChar()).Where(c => c != '\0'));
    }

    public void Clear()
    {
        _keyBuffer.Clear();
    }

    public void PressKey(ConsoleKey key)
    {
        _keyBuffer.Add(key);
        HandleInputKey();
    }

    public void WaitForInput()
    {
        var keyInfo = _consoleDevice.ReadKey(intercept: true);
        _keyBuffer.Add(keyInfo.Key);
        HandleInputKey();
    }

    private void HandleInputKey()
    {
        // 計算匹配的模式數量
        var matchCount = 0;
        IKeyPattern? matchedPattern = null;
        
        foreach (var pattern in _keyPatterns.Keys)
        {
            if (pattern.IsMatch(_keyBuffer))
            {
                matchCount++;
                matchedPattern = pattern;
            }
        }
        
        // 如果只有一個模式匹配，執行對應的操作
        if (matchCount == 1 && matchedPattern != null)
        {
            _keyPatterns[matchedPattern].Invoke();
            _keyBuffer.Clear();
        }
        // 如果沒有模式匹配，但緩衝區已經達到一定長度，清除緩衝區
        else if (matchCount == 0 && _keyBuffer.Count >= 3)
        {
            _keyBuffer.Clear();
        }
    }
}