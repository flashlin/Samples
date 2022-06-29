namespace GitCli.Models;

public interface ICommand
{
    bool IsMyCommand(string[] args);
    Task Run();
}