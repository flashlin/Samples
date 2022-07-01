namespace GitCli.Models.ConsoleMixedReality;

public class InputEvent
{
    public bool HasControl { get; set; }
    public bool HasAlt { get; set; }
    public bool HasShift { get; set; }
    public ConsoleKey Key { get; set; }
    public bool Handled { get; set; }
    public char KeyChar { get; set; }
}