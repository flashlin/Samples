using GitCli.Models.ConsoleMixedReality;
using LanguageExt.Common;

namespace GitCli.Models;

public class GitStatusCommand : ICommand
{
	private GitStatusCommandArgs _args;
	private IConsoleWriter _console = new ConsoleWriter();
	
	public bool IsMyCommand(string[] args)
	{
		var p = args.ParseArgs<GitStatusCommandArgs>();
		return p.Match(v =>
		{
			_args = v;
			return true;
		}, _ => false);
	}

	public Task Run()
	{
		_console.WriteLine($"Action='{_args.ActionName}'");
		return Task.CompletedTask;
	}
}