namespace GitCli.Models;

public interface IApplicationWindow
{
    Task Run(string[] args);
}

public interface ITerminalGui : IApplicationWindow
{
    bool Confirm(string title, string message);
    GitRepoInfo GetRepoInfo();
}