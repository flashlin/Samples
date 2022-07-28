namespace T1.ConsoleUiMixedReality;

public class ConsoleElementEvent
{
	public InputEvent InputEvent { get; set; } = InputEvent.Empty;
	public IConsoleElement Element { get; set; } = EmptyElement.Default;
}