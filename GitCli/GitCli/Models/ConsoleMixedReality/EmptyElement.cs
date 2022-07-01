namespace GitCli.Models.ConsoleMixedReality;

public class EmptyElement : IConsoleElement
{
    private readonly IConsoleWriter _console;

    public EmptyElement(IConsoleWriter console)
    {
        _console = console;
    }

    public Func<Rect> GetViewRect
    {
        get { return () => Rect.OfSize(_console.GetSize()); }
    }

    public Character this[Position pos] => Character.Empty;

    public bool OnInput(InputEvent inputEvent)
    {
        return false;
    }
}