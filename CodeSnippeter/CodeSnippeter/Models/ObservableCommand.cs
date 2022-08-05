using System;
using System.Windows.Input;

namespace CodeSnippeter.Models;

public abstract class ObservableCommand : ICommand
{
	public event EventHandler? CanExecuteChanged = delegate { };

	public event EventHandler? CommandExecuted = delegate { };

	public virtual bool CanExecute(object? parameter)
	{
		return true;
	}

	public void Execute(object? parameter)
	{
		ExecuteCommand(parameter);
		CommandExecuted?.Invoke(this, EventArgs.Empty);
	}

	protected abstract void ExecuteCommand(object? parameter);
	protected void RaiseCanExecuteChanged()
	{
		CanExecuteChanged?.Invoke(this, EventArgs.Empty);
	}
}