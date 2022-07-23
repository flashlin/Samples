using GitCli.Models.ConsoleMixedReality;
using T1.Standard.Extensions;

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

		var allCommitList = new ListBox(new Rect
		{
			Width = 30,
			Height = 20,
		}).Setup(x =>
		{
			x.Name = "allCommitList";
		});

		var localChanges = new TextBox()
		{
			Value = "Local Changes",
		};

		var changesList = new ListBox(new Rect
		{
			Width = 20,
			Height = 2,
		}).Setup(x =>
		{
			x.Name = "LocalChanges";
			//var localChanges = x.AddItem(new ListItem()
			//{
			//	Title = "Local Changes",
			//});

			x.AddElement(localChanges);

			localChanges.OnHandle += (sender, evt) =>
			{
				var fileStatus = gitRepoInfo.QueryStatus()
					.ToArray();

				Console.WriteLine("");
			};

			var allCommits = x.AddItem(new ListItem()
			{
				Title = "All Commits",
			});

			allCommits.OnHandle += (sender, evt) =>
			{
				var commits = gitRepoInfo.QueryCommits();
				foreach (var commit in commits)
				{
					allCommitList.AddItem(new ListItem()
					{
						Title = commit.Message,
						Value = commit
					});
				}
				allCommitList.Refresh();
			};
		});

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
					 changesList,
					 new ListBox(new Rect
					 {
						  Left = 0,
						  Top = 3,
						  Width = 20,
						  Height = 10,
					 }).Setup(x =>
					 {
						  x.Name = "branchList";
						  var branches = gitRepoInfo.QueryBranches()
								.OrderByDescending(x => x.IsLocalBranch);
						  foreach (var branch in branches)
						  {
								x.AddItem(new ListItem
								{
									 Title = branch.Name,
									 Value = branch
								});
						  }
					 })
				}
		};

		var layout2 = new VerticalStack()
		{
			Name = "vertical2",
			BackgroundColor = ConsoleColor.DarkGreen,
			Children =
			{
				allCommitList,
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
}

public class ListItem
{
	public string Title { get; set; } = string.Empty;
	public object? Value { get; set; }
}