namespace GitCli.Models.ConsoleMixedReality;

public class Label : IConsoleElement
{
    public Label(Rect rect)
    {
        ViewRect = rect;
    }

    public IConsoleElement? Parent { get; set; }
    public bool IsTab { get; set; }
    public bool Enabled { get; set; }

    public Color Background { get; set; } = ConsoleColor.DarkBlue;

    public Position CursorPosition => Position.Empty;

    public Rect ViewRect { get; set; }
    public string Value { get; set; } = String.Empty;

    public Character this[Position pos]
    {
        get
        {
            if (!ViewRect.Contain(pos))
            {
                return Character.Empty;
            }

            var x = pos.X - ViewRect.Left;
            var text = Value.SubStr(x, 1);
            if (string.IsNullOrEmpty(text))
            {
                return new Character(' ', null, Background);
            }
            return new Character(text[0], null, Background);
        }
    }

    public bool OnInput(InputEvent inputEvent)
    {
        return false;
    }

    public void OnCreate(Rect ofSize)
    {
    }

    public void OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
    {
    }

    public Rect GetSurroundChildrenRect()
    {
	    return ViewRect;
    }
}