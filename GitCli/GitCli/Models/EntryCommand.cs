using GitCli.Models.ConsoleMixedReality;

namespace GitCli.Models;

public class EntryCommand : IModelCommand
{
	private readonly (string, Action) _listener;
	public EntryCommand((string value, Action handler) listener)
	{
		_listener = listener;
	}
	public bool CanExecute(ConsoleElementEvent evt)
	{
		return true;
	}
	public void Execute(ConsoleElementEvent evt)
	{
		var consoleElement = evt.Element;
		if (consoleElement.Value == _listener.Item1)
		{
			_listener.Item2();
		}
	}
}