namespace GitCli.Models;

public interface ICliWindow
{
    Task Run(string[] args);
}