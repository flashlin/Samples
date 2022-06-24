using Terminal.Gui;

namespace GitCli.Models;

public class GitChanges : IMenuItem
{
    private View _listView;
    private IApplicationWindow _applicationWindow;

    public GitChanges(IApplicationWindow applicationWindow, View listView)
    {
        _applicationWindow = applicationWindow;
        _listView = listView;
    }

    public string Title { get; set; } = "Changes";

    public void Execute()
    {
        // if (_applicationWindow.Confirm("Changes", "123"))
        // {
        //     var repoInfo = _applicationWindow.GetRepoInfo();
        //     var fileStatus = repoInfo.QueryStatus().ToList();
        //     Title = $"Changes({fileStatus.Count})";
        //     _listView.SetNeedsDisplay();
        // }
        
        var repoInfo = _applicationWindow.GetRepoInfo();
        var fileStatus = repoInfo.QueryStatus().ToList();
        repoInfo.Status.SetValue(fileStatus);
        Title = $"Changes({fileStatus.Count})";
    }

    public override string ToString()
    {
        return Title;
    }
}