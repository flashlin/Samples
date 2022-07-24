using GitCli.Models.ConsoleMixedReality;

namespace GitCli.Models;

public interface IModelCommand
{
	void Execute(ConsoleElementEvent evt);
}