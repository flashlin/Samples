using T1.ConsoleUiMixedReality;

namespace GitCli.Models;

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
		model.ChangesList.Init(new[]{
			new ListItem()
			{
				Title = "Local Changes",
			},
			new ListItem()
			{
				Title = "All Commits",
			},
		});

		model.CompareList.Init(new []{
			new ListItem()
			{
				Title = "compare1"
			},
		});

		model.ChangedFilesList.Init(new []{
			new ListItem()
			{
				Title = "file1"
			},
		});


		GetBranchList(gitRepoInfo, model);

		//
		var mainStack = new HorizontalStack()
		{
			Children =
			{
				CreateLayout1(consoleSize, model),
				CreateLayout2(model),
				new DropdownListBox()
				{
					DataContext = model.AllCommitList
				},
			}
		};

		_consoleManager.Content = mainStack;
		_consoleManager.Start();

		return Task.CompletedTask;
	}

	private static VerticalStack CreateLayout2(MainModel model)
	{
		return new VerticalStack()
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
					Command = model.ACommitCommand,
				},
				new HorizontalStack()
				{
					Name = "changedFilesLayout",
					DesignRect = new Rect()
					{
						Width = 60,
						Height = 10
					},
					Children =
					{
						new ListBox(new Rect()
						{
							Width = 30,
							Height = 10,
						})
						{
							DataContext = model.ChangesList,
						},
						new ListBox(new Rect()
						{
							Width = 30,
							Height = 10,
						})
						{
							DataContext = model.CompareList,
						},
					}
				}
			}
		};
	}

	private static VerticalStack CreateLayout1(Size consoleSize, MainModel model)
	{
		return new VerticalStack()
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

