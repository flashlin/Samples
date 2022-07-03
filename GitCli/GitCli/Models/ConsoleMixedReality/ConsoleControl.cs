namespace GitCli.Models.ConsoleMixedReality;

public abstract class ConsoleControl : IConsoleElement
{
    public IConsoleWriter VConsole { get; private set; }

    public virtual Character this[Position pos] => Character.Empty;

    public virtual Position CursorPosition { get; } = Position.Empty;
    public virtual Rect ViewRect { get; set; } = Rect.Empty;
    
    public List<IConsoleElement> Children { get; set; } = new List<IConsoleElement>();
	
    public virtual bool OnInput(InputEvent inputEvent)
    {
        return false;
    }

    public virtual void OnCreated()
    {
    }

    void IConsoleElement.SetConsoleInstance(IConsoleWriter console)
    {
        VConsole = console;
        foreach (var child in Children)
        {
            child.SetConsoleInstance(console);
        }
    }
}