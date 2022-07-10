using GitCli.Models.ConsoleMixedReality;
using T1.Standard.Extensions;

namespace GitCli.Models;


public class Bind<T>
{
	public T Value { get; set; }
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

		var branchStackLayout = new VerticalStack()
		{
			ViewRect = new Rect()
			{
				Width = 20,
				Height = 20,
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
								  allCommitList.Value.AddItem(new ListItem()
								  {
									  Title = commit.Message,
									  Value = commit
								  });
							  }
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
					Width = 20,
					Height = 40
				}).Setup(x =>
				{
					allCommitList.Value = x;
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
		//
		// do
		// {
		//     var lineCommand = Console.ReadLine();
		//     var lineArgs = lineCommand.ParseCommandArgsLine().ToArray();
		//
		//     var commands = new ICommand[]
		//     {
		//         _serviceProvider.GetService<GitStatusCommand>()!,
		//     };
		//     var cmd = commands.FirstOrDefault(x => x.IsMyCommand(lineArgs));
		//     if (cmd == null)
		//     {
		//         _console.SetForegroundColor(ConsoleColor.Red);
		//         _console.WriteLine($"Unknown command: {lineCommand}");
		//         _console.ResetColor();
		//         continue;
		//     }
		//     await cmd.Run();
		// } while (true);
	}
}

public class ListItem
{
	public string Title { get; set; }
	public object Value { get; set; }
}