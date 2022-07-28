namespace T1.ConsoleUiMixedReality;

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