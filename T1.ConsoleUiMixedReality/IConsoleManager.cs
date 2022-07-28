namespace T1.ConsoleUiMixedReality;

public interface IConsoleManager
{
	IConsoleWriter Console { get; }
	IConsoleElement FocusedElement { get; set; }
	Color HighlightBackgroundColor1 { get; set; }
	Color HighlightBackgroundColor2 { get; set; }
	Color InputBackgroundColor { get; set; }
	ConsoleInputObserver InputObserver { get; }
	Color ViewBackgroundColor { get; set; }
	bool FirstSetFocusElement(IConsoleElement element);
	void SetFocusElementOrChild(IConsoleElement element, IConsoleElement child);
}