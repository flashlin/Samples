namespace GitCli.Models.ConsoleMixedReality;

public class EmptyElement : IConsoleElement
{
    public Rect ViewRect { get; set; } = Rect.Empty;
    public IConsoleElement? Parent { get; set; }

    public bool OnInput(InputEvent inputEvent)
    {
        return false;
    }

    public void OnCreated(IConsoleWriter console)
    {
    }

    public Character this[Position pos] => Character.Empty;

    public Position CursorPosition => Position.Empty;
}