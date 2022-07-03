namespace GitCli.Models.ConsoleMixedReality;

public class EmptyElement : ConsoleControl
{
    public Rect ViewRect { get; set; } = Rect.Empty;

    public override Position CursorPosition => new Position(0, 0);
    
}