namespace GitCli.Models.ConsoleMixedReality;

public class EmptyElement : IConsoleEditableElement
{
    private int _editIndex = 0;
    public Rect ViewRect { get; set; } = Rect.Empty;
    public IConsoleElement? Parent { get; set; }

    public bool OnInput(InputEvent inputEvent)
    {
        return false;
    }

    public void OnCreated(IConsoleWriter console)
    {
    }

    public void OnBubbleEvent(InputEvent inputEvent)
    {
        Parent?.OnBubbleEvent(inputEvent);
    }

    public Character this[Position pos] => Character.Empty;

    public Position CursorPosition => Position.Empty;
    public int EditIndex
    {
        get => _editIndex;
        set => _editIndex = 0;
    }
}