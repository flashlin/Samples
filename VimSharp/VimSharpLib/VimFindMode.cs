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

    public VimFindMode(VimEditor instance)
    {
        Instance = instance;
        _keyHandler = new KeyHandler(instance.Console);
        InitializeKeyHandler();
    }

    public VimEditor Instance { get; }

    private void InitializeKeyHandler()
    {
        _keyHandler.InitializeKeyPatterns(new Dictionary<IKeyPattern, Action<List<ConsoleKeyInfo>>>
        {
            { new AnyKeyPattern(), HandleAnyKeyInput }
        });
    }

    private void HandleAnyKeyInput(List<ConsoleKeyInfo> keys)
    {
        if (_findChar == null)
        {
            _findChar = keys[0].KeyChar;
            return;
        }
        _keyBuffer.Add(keys[0].KeyChar);
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
        // 設置控制台游標位置
        outputBuffer.Append($"\x1b[{Instance.Context.CursorY+1};{Instance.Context.CursorX+1}H");
        // 顯示游標
        outputBuffer.Append("\x1b[?25h");
        // 顯示方塊游標
        outputBuffer.Append("\x1b[2 q");
    }

    public void Render(ColoredChar[,] screenBuffer)
    {
        if (_findChar == null) return;

        int startX = Instance.Context.ViewPort.X + Instance.Context.GetLineNumberWidth();
        int startY = Instance.Context.ViewPort.Y;
        int endY = Instance.Context.ViewPort.Bottom - Instance.Context.StatusBarHeight;
        int endX = Instance.Context.ViewPort.Right;

        _currentLabelCount = 0;
        _currentLabel = ['A'];
        for (int y = startY; y <= endY; y++)
        {
            for (int x = startX; x <= endX; x++)
            {
                if (screenBuffer[y, x].Char == _findChar)
                {
                    // 放置標籤
                    PlaceLabel(screenBuffer, y, x);
                }
            }
        }
    }

    private void PlaceLabel(ColoredChar[,] screenBuffer, int y, int x)
    {
        int screenWidth = Instance.Context.ViewPort.Right - Instance.Context.ViewPort.X - Instance.Context.GetLineNumberWidth();
        
        // 檢查左側是否有足夠空間
        bool hasLeftSpace = x - _labelLength >= Instance.Context.ViewPort.X + Instance.Context.GetLineNumberWidth();
        // 檢查右側是否有足夠空間
        bool hasRightSpace = x + _labelLength <= screenWidth;

        // 如果左側有空間，優先放在左側
        if (hasLeftSpace)
        {
            for (int i = 0; i < _labelLength; i++)
            {
                screenBuffer[y, x - _labelLength + i] = new ColoredChar(_currentLabel[i], ConsoleColor.DarkBlue, ConsoleColor.White);
            }
        }
        // 如果左側沒有空間但右側有空間，放在右側
        else if (hasRightSpace)
        {
            for (int i = 0; i < _labelLength; i++)
            {
                screenBuffer[y, x + i] = new ColoredChar(_currentLabel[i], ConsoleColor.DarkBlue, ConsoleColor.White);
            }
        }
        // 如果兩側都沒有空間，不顯示標籤
        else
        {
            return;
        }

        // 更新標籤
        UpdateLabel();
    }

    private void UpdateLabel()
    {
        _currentLabelCount++;

        // 如果當前標籤已經用完（例如到了 'Z'），增加標籤長度
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

        // 更新當前標籤
        int remainder = _currentLabelCount;
        for (int i = _labelLength - 1; i >= 0; i--)
        {
            int value = remainder % 26;
            remainder /= 26;
            _currentLabel[i] = (char)('A' + value);
        }
    }
} 