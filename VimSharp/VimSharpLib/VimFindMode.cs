using System.Text;
using System.Collections.Generic;

namespace VimSharpLib;

public class VimFindMode : IVimMode
{
    private readonly KeyHandler _keyHandler;
    private char? _findChar;
    private int _labelLength = 1;
    private int _currentLabelCount = 0;
    private List<char> _currentLabel = new();
    private List<char> _keyBuffer = new();
    private Dictionary<string, MatchLabel> _matches = new();
    private IVimMode _vimHomeMode;
    private Action<bool>? _backHandler;
    private int _screenWidth;
    private int _lineNumberWidth;
    private readonly ConsoleColor _labelForegroundColor = ConsoleColor.Yellow;
    private readonly ConsoleColor _labelBackgroundColor = ConsoleColor.DarkBlue;

    public VimFindMode(VimEditor instance, IVimMode vimHomeMode)
    {
        _vimHomeMode = vimHomeMode;
        Instance = instance;
        _keyHandler = new KeyHandler(instance.Console);
        InitializeKeyHandler();
        UpdateScreenMetrics();
    }

    public VimEditor Instance { get; set; }

    private void UpdateScreenMetrics()
    {
        _screenWidth = Instance.Context.ViewPort.Right - Instance.Context.ViewPort.X - Instance.Context.GetLineNumberWidth();
        _lineNumberWidth = Instance.Context.GetLineNumberWidth();
    }

    private void InitializeKeyHandler()
    {
        _keyHandler.InitializeKeyHandlers(new Dictionary<IKeyPattern, Action<List<ConsoleKeyInfo>>>
        {
            { new ConsoleKeyPattern(ConsoleKey.Escape), HandleEscapeKey },
            { new AnyKeyPattern(), HandleAnyKeyInput },
        });
    }

    private void HandleEscapeKey(List<ConsoleKeyInfo> keys)
    {
        CancelJumpTo();
    }

    private void HandleAnyKeyInput(List<ConsoleKeyInfo> keys)
    {
        if (_findChar == null)
        {
            _findChar = keys[0].KeyChar;
            if (!IsExistsLabel(_findChar.Value))
            {
                _findChar = null;
                CancelJumpTo();
            }
            return;
        }

        _keyBuffer.Add(keys[0].KeyChar);
        var keyString = new string(_keyBuffer.ToArray()).ToLower();

        // 使用字典直接查找，O(1) 操作
        if (_matches.TryGetValue(keyString, out var match))
        {
            Instance.Context.CursorX = match.X;
            Instance.Context.CursorY = match.Y;
            Instance.Mode = _vimHomeMode;
            _backHandler?.Invoke(true);
            return;
        }

        // 檢查是否有相同長度的匹配
        var matchCount = _matches.Count(m => m.Key.Length == _keyBuffer.Count);
        if (matchCount == 0)
        {
            CancelJumpTo();
            return;
        }
        
        if (_labelLength > _keyBuffer.Count)
        {
            return;
        }

        CancelJumpTo();
    }

    private void CancelJumpTo()
    {
        _findChar = null;
        _keyBuffer.Clear();
        Instance.Mode = _vimHomeMode;
        _backHandler?.Invoke(false);
    }

    public void PressKey(ConsoleKeyInfo keyInfo)
    {
        _keyHandler.PressKey(keyInfo);
    }

    public void WaitForInput()
    {
        _keyHandler.WaitForInput();
    }

    public void AfterRender(StringBuilder outputBuffer)
    {
        outputBuffer.Append($"\x1b[{Instance.Context.CursorY+1};{Instance.Context.CursorX+1}H");
        outputBuffer.Append("\x1b[?25h");
        outputBuffer.Append("\x1b[2 q");
    }

    public void Render(ColoredCharScreen screenBuffer)
    {
        if (_findChar == null)
        {
            return;
        }

        // 只在需要時更新螢幕度量
        if (_screenWidth != Instance.Context.ViewPort.Right - Instance.Context.ViewPort.X - Instance.Context.GetLineNumberWidth())
        {
            UpdateScreenMetrics();
        }

        // 只在第一次渲染時初始化
        if (_matches.Count == 0)
        {
            _currentLabelCount = 0;
            _currentLabel = ['A'];
            _matches.Clear();
        }

        var startX = Instance.Context.ViewPort.X + _lineNumberWidth;
        var startY = Instance.Context.ViewPort.Y;
        var endY = Instance.Context.ViewPort.Bottom - Instance.Context.StatusBarHeight;
        var endX = Instance.Context.ViewPort.Right;
        for (var y = startY; y <= endY; y++)
        {
            for (var x = startX; x <= endX; x++)
            {
                if (screenBuffer[y, x].Char == _findChar)
                {
                    PlaceLabel(screenBuffer, y, x);
                }
            }
        }
    }

    private bool IsExistsLabel(char findChar)
    {
        var startX = Instance.Context.ViewPort.X + _lineNumberWidth;
        var startY = Instance.Context.ViewPort.Y;
        var endY = Instance.Context.ViewPort.Bottom - Instance.Context.StatusBarHeight;
        var endX = Instance.Context.ViewPort.Right;
        for (var y = startY; y <= endY; y++)
        {
            var textY = Instance.Context.ComputeTextY(y);
            for (var x = startX; x <= endX; x++)
            {
                var textX = Instance.Context.ComputeTextX(x);
                if (textY < Instance.Context.Texts.Count)
                {
                    var line = Instance.Context.Texts[textY];
                    if (textX < line.Chars.Length)
                    {
                        var character = line.Chars[textX];
                        if (character.Char == findChar)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    private void PlaceLabel(ColoredCharScreen screenBuffer, int y, int x)
    {
        // 檢查左側是否有足夠空間
        var hasLeftSpace = x - _labelLength >= Instance.Context.ViewPort.X + _lineNumberWidth;
        // 檢查右側是否有足夠空間
        var hasRightSpace = x + _labelLength <= _screenWidth;

        // 如果左側有空間，優先放在左側
        if (hasLeftSpace)
        {
            for (int i = 0; i < _labelLength; i++)
            {
                screenBuffer[y, x - _labelLength + i] = new ColoredChar(_currentLabel[i], _labelForegroundColor, _labelBackgroundColor);
            }
            
            // 檢查標記前面是否為中文字元，如果是，則將其覆蓋為空格
            var prevIndex = x - _labelLength - 1;
            if (prevIndex > 0 && screenBuffer[y, prevIndex].Char.GetCharWidth() > 1)
            {
                screenBuffer[y, prevIndex] = new ColoredChar(' ', _labelForegroundColor, _labelBackgroundColor);
            }
            
            var label = new string(_currentLabel.ToArray());
            _matches[label.ToLower()] = new MatchLabel(x, y, label);
        }
        // 如果左側沒有空間但右側有空間，放在右側
        else if (hasRightSpace)
        {
            for (int i = 0; i < _labelLength; i++)
            {
                screenBuffer[y, x + i] = new ColoredChar(_currentLabel[i], _labelForegroundColor, _labelBackgroundColor);
            }
            var label = new string(_currentLabel.ToArray());
            _matches[label.ToLower()] = new MatchLabel(x, y, label);
        }
        // 如果兩側都沒有空間，不顯示標籤
        else
        {
            return;
        }

        UpdateLabel();
    }

    private void UpdateLabel()
    {
        _currentLabelCount++;

        if (_currentLabelCount >= Math.Pow(26, _labelLength))
        {
            _labelLength++;
            _currentLabelCount = 0;
            _currentLabel = new List<char>();
            for (int i = 0; i < _labelLength; i++)
            {
                _currentLabel.Add('A');
            }
            return;
        }

        int remainder = _currentLabelCount;
        for (int i = _labelLength - 1; i >= 0; i--)
        {
            int value = remainder % 26;
            remainder /= 26;
            _currentLabel[i] = (char)('A' + value);
        }
    }

    public void SetBackHandler(Action<bool> backHandler)
    {
        _backHandler = backHandler;
    }
} 