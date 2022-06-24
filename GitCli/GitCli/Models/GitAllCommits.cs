namespace GitCli.Models;

public class GitAllCommits : IMenuItem
{
    public string Title { get; set; } = "All Commits";

    public void Execute()
    {
        throw new NotImplementedException();
    }

    public override string ToString()
    {
        return Title;
    }
}