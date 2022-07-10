using GitCli.Models.ConsoleMixedReality;
using T1.Standard.Extensions;

namespace GitCli.Models;


public class Bind<T>
{
	public T? Value { get; private set; }

	public List<Action<T>> SetupList = new List<Action<T>>();

	public void Setup(Action<T> fn)
	{
		if (Value == null)
		{
			SetupList.Add(fn);
			return;
		}
		fn(Value);
	}

	public void SetValue(T value)
	{
		Value = value;
		foreach (var setup in SetupList)
		{
			setup(value);
		}
	}
}

public class ConsoleWindow : IConsoleWindow
{
	private readonly IServiceProvider _serviceProvider;
	private readonly ConsoleWriter _console;
	private ConsoleManager _consoleManager;
	private IGitRepoAgent _gitRepoAgent;

	public ConsoleWindow(IServiceProvider serviceProvider, ConsoleManager consoleManager,
		 IGitRepoAgent gitRepoAgent)
	{
		_gitRepoAgent = gitRepoAgent;
		_consoleManager = consoleManager;
		_serviceProvider = serviceProvider;
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

		var c1 = new StackChildren();
		for (var i = 0; i < 5; i++)
		{
			c1.Add(new TextBox(Rect.Empty)
			{
				Value = $"A{i}"
			});
		}
		var t1 = new TableRow(new Rect
		{
			Left = 0,
			Top = 0,
			Width = 30,
			Height = 1
		}, c1);

		_consoleManager.Content = mainStack;
		//_consoleManager.Content = t1;
		_consoleManager.Start();

		return Task.CompletedTask;
	}
}

public class ListItem
{
	public string? Title { get; set; }
	public object? Value { get; set; }
}