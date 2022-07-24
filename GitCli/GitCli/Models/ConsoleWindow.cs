using System.Collections.ObjectModel;
using GitCli.Models.ConsoleMixedReality;
using T1.Standard.Extensions;

namespace GitCli.Models;

public class MainModel
{
	public MainModel()
	{
		LocalChangesCommand = new EntryCommand(("All Commits", OnHandleAllChanges));
	}

	public GitRepoInfo RepoInfo { get; set; }
	public NotifyCollection<ListItem> ChangesList { get; set; } = new();
	public NotifyCollection<ListItem> BranchList { get; set; } = new();
	public NotifyCollection<ListItem> AllCommitList { get; set; } = new();
	public IModelCommand LocalChangesCommand { get; set; }

	private void OnHandleAllChanges()
	{
		var commits = RepoInfo.QueryCommits();
		foreach (var commit in commits)
		{
			AllCommitList.Adding(new ListItem()
			{
				Title = commit.Message,
				Value = commit
			});
		}
		AllCommitList.Notify();
	}
}


public class EntryCommand : IModelCommand
{
	private readonly (string, Action) _listener;
	public EntryCommand((string value, Action handler) listener)
	{
		_listener = listener;
	}
	public bool CanExecute(ConsoleElementEvent evt)
	{
		return true;
	}
	public void Execute(ConsoleElementEvent evt)
	{
		var consoleElement = evt.Element;
		if (consoleElement.Value == _listener.Item1)
		{
			_listener.Item2();
		}
	}
}

public interface IModelCommand
{
	void Execute(ConsoleElementEvent evt);
}

public class ConsoleWindow : IConsoleWindow
{
	private readonly ConsoleWriter _console;
	private readonly ConsoleManager _consoleManager;
	private readonly IGitRepoAgent _gitRepoAgent;

	public ConsoleWindow(ConsoleManager consoleManager,
		 IGitRepoAgent gitRepoAgent)
	{
		_gitRepoAgent = gitRepoAgent;
		_consoleManager = consoleManager;
		_console = new ConsoleWriter();
	}

	public Task Run(string[] args)
	{
		var gitRepoInfo = _gitRepoAgent.OpenRepoFolder("D:/VDisk/Github/Codewars");
		var consoleSize = _console.GetSize();
		var model = new MainModel();
		model.RepoInfo = gitRepoInfo;
		model.ChangesList.Adding(new ListItem()
		{
			Title = "Local Changes",
		});
		model.ChangesList.Adding(new ListItem()
		{
			Title = "All Commits",
		});
		model.ChangesList.Notify();

		GetBranchList(gitRepoInfo, model);

		//

		var changedFilesList = new ListBox(new Rect()
		{
			Width = 30,
			Height = 10,
		});

		changedFilesList.AddItem(new ListItem()
		{
			Title = "file1"
		});

		var compareList = new ListBox(new Rect()
		{
			Width = 30,
			Height = 10,
		});

		compareList.AddItem(new ListItem()
		{
			Title = "compare1"
		});

		var changedFilesLayout = new HorizontalStack()
		{
			Name = "changedFilesLayout",
			DesignRect = new Rect()
			{
				Width = 60,
				Height = 10
			},
			Children =
			{
				changedFilesList,
				compareList,
			}
		};

		var layout1 = new VerticalStack()
		{
			Name = "LeftVertical1",
			DesignRect = new Rect()
			{
				Width = 20,
				Height = consoleSize.Height,
			},
			BackgroundColor = ConsoleColor.DarkMagenta,
			Children =
				{
					 new ListBox(new Rect
					 {
						 Width = 20,
						 Height = 2,
					 })
					 {
						 Name = "LocalChanges",
						 DataContext = model.ChangesList,
						 Command = model.LocalChangesCommand
					 },
					 new ListBox(new Rect
					 {
						  Left = 0,
						  Top = 3,
						  Width = 20,
						  Height = 10,
					 })
					 {
						  Name = "branchList",
						  DataContext = model.BranchList
					 },
				}
		};

		var layout2 = new VerticalStack()
		{
			Name = "vertical2",
			BackgroundColor = ConsoleColor.DarkGreen,
			Children =
			{
				new ListBox(new Rect
				{
					Width = 30,
					Height = 20,
				})
				{
					Name = "allCommitList",
					DataContext = model.AllCommitList,
				},
				changedFilesLayout
			}
		};

		var mainStack = new HorizontalStack()
		{
			Children =
			{
				layout1,
				layout2
			}
		};

		_consoleManager.Content = mainStack;
		_consoleManager.Start();

		return Task.CompletedTask;
	}

	private static void GetBranchList(GitRepoInfo gitRepoInfo, MainModel model)
	{
		var branches = gitRepoInfo.QueryBranches()
			.OrderByDescending(x => x.IsLocalBranch);
		foreach (var branch in branches)
		{
			model.BranchList.Adding(new ListItem
			{
				Title = branch.Name,
				Value = branch
			});
		}

		model.BranchList.Notify();
	}
}

public class ListItem
{
	public string Title { get; set; } = string.Empty;
	public object? Value { get; set; }
}