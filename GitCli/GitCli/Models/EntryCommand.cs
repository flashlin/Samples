using GitCli.Models.ConsoleMixedReality;

namespace GitCli.Models;

public class EntryCommand : IModelCommand
{
	private readonly string _value;
	private readonly Action _handler;

	public EntryCommand(string value, Action handler)
	{
		_handler = handler;
		_value = value;
	}
	public bool CanExecute(ConsoleElementEvent evt)
	{
		return evt.Element.Value == _value;
	}
	public void Execute(ConsoleElementEvent evt)
	{
		_handler();
	}
}

public class ExecuteCommand : IModelCommand
{
	private readonly Action<ConsoleElementEvent> _handler;

	public ExecuteCommand(Action<ConsoleElementEvent> handler)
	{
		_handler = handler;
	}
	public bool CanExecute(ConsoleElementEvent evt)
	{
		return true;
	}
	public void Execute(ConsoleElementEvent evt)
	{
		_handler(evt);
	}
}

public static class ModelCommandExtension
{
	public static void Raise(this IModelCommand? command, ConsoleElementEvent evt)
	{
		if (command == null)
		{
			return;
		}
		if (command.CanExecute(evt))
		{
			command.Execute(evt);
		}
	}
}