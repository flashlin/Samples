namespace VimSharpLib;

public class KeyHandler
{
    private readonly IConsoleDevice _consoleDevice;
    private Dictionary<IKeyPattern, Action<List<ConsoleKeyInfo>>> _keyPatterns = new();
    private readonly List<ConsoleKeyInfo> _keyBuffer = new();
    
    public KeyHandler(IConsoleDevice consoleDevice)
    {
        _consoleDevice = consoleDevice;
    }

    public void InitializeKeyPatterns(Dictionary<IKeyPattern, Action<List<ConsoleKeyInfo>>> keyPatterns)
    {
        _keyPatterns = keyPatterns;
    }
    
    public string GetKeyBufferString()
    {
        return string.Join("", _keyBuffer.Select(k => k.KeyChar).Where(c => c != '\0'));
    }

    public void Clear()
    {
        _keyBuffer.Clear();
    }

    public void PressKey(ConsoleKeyInfo keyInfo)
    {
        _keyBuffer.Add(keyInfo);
        HandleInputKey();
    }

    public void WaitForInput()
    {
        var keyInfo = _consoleDevice.ReadKey(intercept: false);
        _keyBuffer.Add(keyInfo);
        HandleInputKey();
    }

    private void HandleInputKey()
    {
        // 計算匹配的模式數量
        var matchCount = 0;
        IKeyPattern? matchedPattern = null;
        List<IKeyPattern> matchedPatterns = new();
        
        foreach (var pattern in _keyPatterns.Keys)
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
            _keyPatterns[nonAnyKeyPattern].Invoke(_keyBuffer.ToList());
            _keyBuffer.Clear();
            return;
        }
        
        // 如果只有一個模式匹配，執行對應的操作
        if (matchCount == 1 && matchedPattern != null)
        {
            _keyPatterns[matchedPattern].Invoke(_keyBuffer.ToList());
            _keyBuffer.Clear();
        }
        else if (matchCount == 0 && _keyBuffer.Count >= 3)
        {
            _keyBuffer.Clear();
        }
    }
}