namespace GitCli.Models.ConsoleMixedReality;

public class Frame : IConsoleElement
{
    public Frame(Rect viewRect)
    {
        ViewRect = viewRect;
    }

    private IConsoleElement? _focus;

    public Rect ViewRect { get; set; }

    public IConsoleElement? Parent { get; set; }

    public List<IConsoleElement> Children { get; set; } = new List<IConsoleElement>();

    public Position CursorPosition
    {
        get
        {
            GetFocusedControl();

            if (_focus != null)
            {
                return _focus.CursorPosition;
            }

            return ViewRect.BottomRightCorner;
        }
    }

    private IConsoleElement? GetFocusedControl()
    {
        _focus ??= Children.FirstOrDefault();
        return _focus;
    }

    public Character this[Position pos]
    {
        get
        {
            if (!ViewRect.Contain(pos))
            {
                return Character.Empty;
            }

            var character = new Character(' ', null, ConsoleColor.DarkGray);
            foreach (var child in Children)
            {
                var ch = child[pos];
                if (!ch.IsEmpty)
                {
                    character = ch;
                }
            }

            return character;
        }
    }

    public void OnCreate(IConsoleManager manager)
    {
        var viewRect = ViewRect.Init(() => Rect.OfSize(manager.Console.GetSize()));
        foreach (var (child, idx) in Children.Select((val, idx) => (val, idx)))
        {
            if (idx == 0)
            {
                _focus = child;
            }

            child.Parent = this;
            child.ViewRect = new Rect
            {
                Left = viewRect.Left + child.ViewRect.Left,
                Top = viewRect.Top + child.ViewRect.Top,
                Width = child.ViewRect.Width,
                Height = child.ViewRect.Height,
            };
            child.OnCreate(manager);
        }
    }

    public bool OnInput(InputEvent inputEvent)
    {
        if (_focus == null)
        {
            return false;
        }

        return _focus.OnInput(inputEvent);
    }

    public void OnBubbleEvent(InputEvent inputEvent)
    {
        if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
        {
            _focus = GetFocusedControl();
            if (_focus != null)
            {
                var idx = Children.FindIndex(x => x == _focus);
                idx = Math.Min(idx - 1, 0);
                _focus = Children[idx];
                return;
            }

            Parent?.OnBubbleEvent(inputEvent);
            return;
        }

        if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
        {
            _focus = GetFocusedControl();
            if (_focus != null)
            {
                var idx = Children.FindIndex(x => x == _focus);
                idx = Math.Min(idx + 1, Children.Count - 1);
                _focus = Children[idx];
                return;
            }
        }

        Parent?.OnBubbleEvent(inputEvent);
    }
}