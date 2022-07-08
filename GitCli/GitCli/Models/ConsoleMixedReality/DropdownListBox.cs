using System.Collections.ObjectModel;

namespace GitCli.Models.ConsoleMixedReality;

public class DropdownListBox : IConsoleElement
{
    private bool _isSelectedMode = false;
    private IConsoleManager _manager;
    private TextBox _textBox;
    private ListBox _listBox;
    private bool _isSelectMode = false;

    public DropdownListBox(Rect rect)
    {
        ViewRect = rect;

        _textBox = new TextBox(new Rect()
        {
            Left = rect.Left,
            Top = rect.Top,
            Width = rect.Width,
            Height = 1,
        });
        _listBox = new ListBox(new Rect()
        {
            Left = rect.Left,
            Top = rect.Top + 1,
            Width = rect.Width,
            Height = rect.Height - 1
        });
    }

    public IConsoleElement? Parent { get; set; }
    public bool IsTab { get; set; } = true;

    public Color BackgroundColor { get; set; } = ConsoleColor.Blue;

    public Position CursorPosition
    {
        get
        {
            if (_isSelectMode)
            {
                return _listBox.CursorPosition;
            }

            return _textBox.CursorPosition;
        }
    }

    public Rect ViewRect { get; set; }

    public Character this[Position pos]
    {
        get
        {
            if (!ViewRect.Contain(pos))
            {
                return Character.Empty;
            }

            var y = pos.Y - ViewRect.Top;
            if (y == 0)
            {
                return _textBox[pos];
            }

            if (!_isSelectedMode)
            {
                return Character.Empty;
            }

            return _listBox[pos];
        }
    }

    public bool OnInput(InputEvent inputEvent)
    {
        if (_isSelectedMode)
        {
            return _listBox.OnInput(inputEvent);
        }

        return _textBox.OnInput(inputEvent);
    }

    public void OnCreate(IConsoleManager manager)
    {
        _manager = manager;
        var y = ViewRect.Top;
        _textBox.Parent = this;
        _listBox.Parent = this;
    }

    public void OnBubbleEvent(InputEvent inputEvent)
    {
    }
}