namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleElementEvent
{
	public InputEvent InputEvent { get; init; } = InputEvent.Empty;
	public IConsoleElement Element { get; init; } = EmptyElement.Default;
}