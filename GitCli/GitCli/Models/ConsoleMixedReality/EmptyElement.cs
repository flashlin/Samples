namespace GitCli.Models.ConsoleMixedReality;

public class EmptyElement : IConsoleElement
{
    public Rect ViewRect { get; set; } = Rect.Empty;
    public bool OnInput(InputEvent inputEvent)
    {
        return false;
    }

    public void OnCreated()
    {
    }

    public Character this[Position pos] => Character.Empty;

    public virtual Position CursorPosition => Position.Empty;

}