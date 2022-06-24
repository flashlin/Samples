using Terminal.Gui;

namespace GitCli.Models;

public class ApplicationWindow : IApplicationWindow
{
    private GitChanges _gitChanges;
    private List<IMenuItem> _workspaceViewMenus;
    private GitAllCommits _gitAllCommits;
    private GitRepoInfo _gitRepoInfo;

    public void Run()
    {
        //new Example().Sample();

        Application.Init();
        var top = Application.Top;

        var workspace = AddWorkspaceWindow(top);

        workspace.GetCurrentHeight(out var workspaceHeight);
        var repositoryWin = new Window("Repository")
        {
            X = 0,
            Y = workspaceHeight + 1,
            Width = Dim.Fill(),
            Height = Dim.Fill()
        };
        top.Add(repositoryWin);

        workspace.GetCurrentWidth(out var workspaceWidth);
        var unstagedWin = new Window("unstaged")
        {
            X = workspace.X + workspaceWidth + 1,
            Y = 1,
            Width = Dim.Fill(),
            Height = Dim.Fill()
        };
        top.Add(unstagedWin);

        AddMenuBar(top);

        Application.Run();
        Application.Shutdown();
    }

    private Window AddWorkspaceWindow(Toplevel top)
    {
        var workspace = new Window("Workspace")
        {
            X = 0,
            Y = 1,
            Width = Dim.Percent(30),
            Height = Dim.Percent(30)
        };
        top.Add(workspace);
        AddWorkSpaceMenu(workspace);
        return workspace;
    }

    private void AddMenuBar(Toplevel top)
    {
        var menu = new MenuBar(new MenuBarItem[]
        {
            new MenuBarItem("_File", new MenuItem[]
            {
                new MenuItem("Clone...", "Clone Repository", null),
                new MenuItem("_Open Repository...", "Open Repository", HandleOpenRepository),
                new MenuItem("_Exit", "", () =>
                {
                    if (Quit()) top.Running = false;
                })
            }),
            new MenuBarItem("_View", new MenuItem[]
            {
                new MenuItem("Show Uncommitted Changes", "", null),
            }),
            new MenuBarItem("_Repository", new MenuItem[]
            {
                new MenuItem("Refresh", "", null),
                new MenuItem("Fetch...", "", null),
                new MenuItem("Pull...", "", null),
                new MenuItem("Push...", "", null),
                new MenuItem("Save Stash...", "", null),
                new MenuItem("New Branch...", "", null),
                new MenuItem("Rebase Merge...", "", null),
            }),
        });
        top.Add(menu);
    }

    private void AddWorkSpaceMenu(Window workspace)
    {
        var workspaceView = new ListView
        {
            X = 0,
            Y = 0,
            Width = Dim.Fill(),
            Height = Dim.Fill(),
        };

        _gitChanges = new GitChanges(this, workspaceView);
        _gitAllCommits = new GitAllCommits();
        _workspaceViewMenus = new List<IMenuItem>()
        {
            _gitChanges,
            _gitAllCommits
        };
        workspaceView.SetSource(_workspaceViewMenus);
        workspace.Add(workspaceView);

        workspaceView.OpenSelectedItem += WorkspaceView_OpenSelectedItem;
    }

    private void WorkspaceView_OpenSelectedItem(ListViewItemEventArgs obj)
    {
        _workspaceViewMenus.First(x => x == obj.Value)
            .Execute();
    }

    public GitRepoInfo GetRepoInfo()
    {
        return _gitRepoInfo;
    }

    public bool Confirm(string title, string message)
    {
        var n = MessageBox.Query(50, 7, title, message, "Yes", "No");
        return n == 0;
    }

    static bool Quit()
    {
        var n = MessageBox.Query(50, 7, "Quit GitCli", "Are you sure you want to quit this GitCli?", "Yes",
            "No");
        return n == 0;
    }

    GitRepoAgent _gitRepoAgent;

    void HandleOpenRepository()
    {
        _gitRepoAgent.OpenRepoFolder();
    }
}