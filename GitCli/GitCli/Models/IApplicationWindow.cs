namespace GitCli.Models;

public interface IApplicationWindow
{
    bool Confirm(string title, string message);
    GitRepoInfo GetRepoInfo();
}