namespace GitCli.Models;

public class GitStatusCommand : ICommand
{
    public bool IsMyCommand(string[] args)
    {
        if (args.Length != 5)
        {
            return false;
        }

        if (args[0] != "b")
        {
            return false;
        }

        return true;
    }

    public Task Run(string[] args)
    {
        var p = args.ParseArgs<GitStatusCommandArgs>()!;
        return Task.CompletedTask;
    }
}