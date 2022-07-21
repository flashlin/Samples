using GitCli.Models.ConsoleMixedReality;
using LanguageExt.Common;

namespace GitCli.Models;

public class GitStatusCommand : ICommand
{
	private GitStatusCommandArgs _args = new();
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
		return Task.CompletedTask;
	}
}