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

		var allCommitList = new Bind<ListBox>();

		var consoleSize = _console.GetSize();

		var branchStackLayout = new VerticalStack()
		{
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
						  Left = 0,
						  Top = 0,
						  Width = 20,
						  Height = 2,
					 }).Setup(x =>
					 {
						  var localChanges = x.AddItem(new ListItem()
						  {
								Title = "Local Changes",
						  });

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
								  allCommitList.Value!.AddItem(new ListItem()
								  {
									  Title = commit.Message,
									  Value = commit
								  });
							  }
							  allCommitList.Value!.Refresh();
						  };

					 }),
					 new ListBox(new Rect
					 {
						  Left = 0,
						  Top = 3,
						  Width = 20,
						  Height = 10,
					 }).Setup(branchList =>
					 {
						  var branches = gitRepoInfo.QueryBranches()
								.OrderByDescending(x => x.IsLocalBranch);
						  foreach (var branch in branches)
						  {
								branchList.AddItem(new ListItem
								{
									 Title = branch.Name,
									 Value = branch
								});
						  }
					 })
				}
		};

		var verticalStack2 = new VerticalStack()
		{
			BackgroundColor = ConsoleColor.DarkGreen,
			Children =
			{
				new ListBox(new Rect
				{
					Left = 0,
					Top = 0,
					Width = 30,
					Height = consoleSize.Height,
				}).Setup(x =>
				{
					x.Name = "allCommitList";
					allCommitList.SetValue(x);
				})
			}
		};

		var mainStack = new HorizontalStack()
		{
			Children =
			{
				branchStackLayout,
				verticalStack2
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