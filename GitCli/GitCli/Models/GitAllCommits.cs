using Terminal.Gui;
using Terminal.Gui.Trees;

namespace GitCli.Models;

public class TreeItem : ITreeNode
{
    public string Text { get; set; }
    public IList<ITreeNode> Children { get; } = new List<ITreeNode>();
    public object Tag { get; set; }
}

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