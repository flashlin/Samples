namespace T1.ConsoleUiMixedReality;

public class EmptyConsoleManager : IConsoleManager
{
	public static EmptyConsoleManager Default = new();
    
	private EmptyConsoleManager()
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

	public void SetFocusElementOrChild(IConsoleElement element, IConsoleElement child)
	{
	}
}