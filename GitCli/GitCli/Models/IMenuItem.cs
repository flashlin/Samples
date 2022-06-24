namespace GitCli.Models;

public interface IMenuItem
{
    string Title { get; set; }
    void Execute();
}