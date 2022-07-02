namespace GitCli.Models.ConsoleMixedReality;

public class VerticalStack : IConsoleElement
{
    private IConsoleElement? _focus;
    
    public VerticalStack()
    {
        GetViewRect = () =>
        {
            var rect = Rect.Empty;
            foreach (var child in Children)
            {
                rect = child.GetViewRect().Surround(rect);
            }
            return rect;
        };
    }

    public Func<Rect> GetViewRect { get; set; }

    public Position CursorPosition
    {
        get
        {
            _focus ??= Children.First();
            return _focus.CursorPosition;
        }
    }

    public List<IConsoleElement> Children { get; set; } = new List<IConsoleElement>();

    public Character this[Position pos]
    {
        get
        {
            var child = Children
                .FirstOrDefault(x => x.GetViewRect().Contain(pos));
            if (child != null)
            {
                return child[pos];
            }
            return Character.Empty;
        }
    }

    public bool OnInput(InputEvent inputEvent)
    {
        var child = Children
            .FirstOrDefault(x => x.OnInput(inputEvent));
        if (child != null)
        {
            return true;
        }
        return false;
    }
}