namespace GitCli.Models.ConsoleMixedReality;

public class EmptyConsoleManager : IConsoleManager
{
    
	public EmptyConsoleManager()
	{
	}

	public IConsoleWriter Console { get; } = new ConsoleWriter();
	public IConsoleElement FocusedElement { get; set; } = EmptyElement.Default;
	public Color HighlightBackgroundColor1 { get; set; }
	public Color HighlightBackgroundColor2 { get; set; }
	public Color InputBackgroundColor { get; set; }
	public ConsoleInputObserver InputObserver { get; } = new();
	public Color ViewBackgroundColor { get; set; }
	public bool FirstSetFocusElement(IConsoleElement element)
	{
		return false;
	}
}