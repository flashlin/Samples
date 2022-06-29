using System.ComponentModel;
using Terminal.Gui;

namespace GitCli.Models;

public interface IObjectNotifyPropertyChanged : INotifyPropertyChanged
{
	void RaisePropertyChanged(string propertyName);
}

public class BaseNotifyObject : IObjectNotifyPropertyChanged
{
	public event PropertyChangedEventHandler? PropertyChanged;

	protected virtual void OnPropertyChanged(PropertyChangedEventArgs args)
	{
		PropertyChanged?.Invoke(this, args);
	}

	public void RaisePropertyChanged(string propertyName)
	{
		OnPropertyChanged(new PropertyChangedEventArgs(propertyName));
	}
}

public class NotifyProperty<T>
{
	private IObjectNotifyPropertyChanged _owner;

	public NotifyProperty(IObjectNotifyPropertyChanged owner, string name, T initialValue)
	{
		_owner = owner;
		Name = name;
		Value = initialValue;
	}

	public string Name { get; }
	public T Value { get; private set; }

	public void SetValue(T newValue)
	{
		if (!newValue.Equals(Value))
		{
			Value = newValue;
			_owner.RaisePropertyChanged(this.Name);
		}
	}
}

public class TerminalGui : ITerminalGui
{
	private GitChanges _gitChanges;
	private List<IMenuItem> _workspaceViewMenus;
	private GitAllCommits _gitAllCommits;
	private GitRepoInfo _gitRepoInfo;
	private IGitRepoAgent _gitRepoAgent;
	private Window _unStagedWindow;
	private Window _workspaceWindow;

	public TerminalGui(IGitRepoAgent gitRepoAgent)
	{
		_gitRepoAgent = gitRepoAgent;
		_gitRepoInfo = _gitRepoAgent.OpenRepoFolder("D:/VDisk/Github/Samples");
	}

	public Task Run(string[] args)
	{
		//new Example().Sample();

		Application.Init();
		var top = Application.Top;

		AddWorkspaceWindow(top);
		var repositoryWindow = new Window("Repository")
		{
			X = 0,
			Y = Pos.Bottom(_workspaceWindow),
			Width = 15,
			Height = Dim.Fill()
		};
		top.Add(repositoryWindow);

		AddUnStagedWindow(top);
		var compareWindow = new Window("compare")
		{
			X = Pos.Right(_unStagedWindow),
			Y = 1,
			Width = Dim.Fill(),
			Height = Dim.Fill()
		};
		top.Add(compareWindow);

		AddMenuBar(top);
		
		//top.Redraw(new Rect(0,0, Application.Top.Bounds.Width, Application.Top.Bounds.Height));
		Application.Run();
		Application.Shutdown();

		return Task.CompletedTask;
	}

	private void AddUnStagedWindow(Toplevel top)
	{
		_unStagedWindow = new Window("unstaged")
		{
			X = Pos.Right(_workspaceWindow),
			Y = 1,
			Width = Dim.Percent(30),
			Height = Dim.Fill()
		};
		top.Add(_unStagedWindow);

		var unStagedTreeView = new TreeView()
		{
			X = 0,
			Y = 0,
			Width = Dim.Fill(),
			Height = Dim.Fill()
		};

		var scrollView = new ScrollView()
		{
			KeepContentAlwaysInViewport = true,
			ShowVerticalScrollIndicator = true,
			ShowHorizontalScrollIndicator = true,
			ContentSize = new Size(200, 150),
			X = 0,
			Y = 0,
			Width = Dim.Fill(),
			Height = Dim.Fill()
		};
		scrollView.Add(unStagedTreeView);
		_unStagedWindow.Add(scrollView);
		//_unStagedWindow.Add(unStagedTreeView);

		_gitRepoInfo.PropertyChanged += (sender, args) =>
		{
			if (args.PropertyName == nameof(_gitRepoInfo.Status))
			{
				unStagedTreeView.ClearObjects();
				unStagedTreeView.AddObjects(_gitRepoInfo.Status.Value.Select(x => new TreeItem()
				{
					Text = x.FilePath
				}));
				//unStagedTreeView.SetNeedsDisplay();
				//scrollView.SetNeedsDisplay();
			}
		};
	}

	private void AddWorkspaceWindow(Toplevel top)
	{
		_workspaceWindow = new Window("Workspace")
		{
			X = 0,
			Y = 1,
			Width = 15,
			Height = Dim.Percent(30)
		};
		top.Add(_workspaceWindow);
		AddWorkSpaceMenu(_workspaceWindow);
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

	private static bool Quit()
	{
		var n = MessageBox.Query(50, 7, "Quit GitCli", "Are you sure you want to quit this GitCli?", "Yes",
			 "No");
		return n == 0;
	}

	private void HandleOpenRepository()
	{
	}
}