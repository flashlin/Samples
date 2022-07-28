namespace T1.ConsoleUiMixedReality;

public interface IModelCommand
{
	void Execute(ConsoleElementEvent evt);
	bool CanExecute(ConsoleElementEvent evt);
}