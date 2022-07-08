using GitCli.Models.ConsoleMixedReality;
using T1.Standard.Extensions;

namespace GitCli.Models;

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
        
        var stackLayout = new VerticalStack()
        {
            ViewRect = new Rect
            {
                Left = 0,
                Top = 1,
                Width = 40,
                Height = 20,
            },
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
                    
                    x.AddItem(new ListItem()
                    {
                        Title = "All Commits",
                    });
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
        
        
        
        

        var listBox = new ListBox(new Rect()
        {
            Left = 0,
            Top = 0,
            Width = 10,
            Height = 5,
        })
        {
            Children =
            {
                new TextBox(Rect.Empty)
                {
                    Value = "1.1234567890"
                },
                new TextBox(Rect.Empty)
                {
                    Value = "2.abcdef"
                },
                new TextBox(Rect.Empty)
                {
                    Value = "3.Flash123"
                },
                new TextBox(Rect.Empty)
                {
                    Value = "4.Jack"
                },
                new TextBox(Rect.Empty)
                {
                    Value = "5.Jack, Mary, Flash"
                },
                new TextBox(Rect.Empty)
                {
                    Value = "6.End"
                },
            }
        };

        var dropdown1 = new DropdownListBox(new Rect()
        {
            Left = 20,
            Top = 1,
            Width = 10,
            Height = 10
        });

        var frame = new Frame(new Rect()
        {
            Left = 0,
            Top = 0,
            Width = 50,
            Height = 20
        })
        {
            Children =
            {
                listBox,
                dropdown1
            }
        };

        _consoleManager.Content = stackLayout;
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