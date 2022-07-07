using GitCli.Models.ConsoleMixedReality;
using T1.Standard.Extensions;

namespace GitCli.Models;

public class ConsoleWindow : IConsoleWindow
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ConsoleWriter _console;
    private ConsoleManager _consoleManager;

    public ConsoleWindow(IServiceProvider serviceProvider, ConsoleManager consoleManager)
    {
        _consoleManager = consoleManager;
        _serviceProvider = serviceProvider;
        _console = new ConsoleWriter();
    }

    public Task Run(string[] args)
    {
        var idTextBox = new TextBox(new Rect
        {
            Left = 10,
            Top = 2,
            Width = 10,
            Height = 1,
        }).Setup(x =>
        {
            x.MaxLength = 5;
            x.TypeCharacter = '*';
        });

        var box = new TextArea(new Rect
        {
            Left = 10,
            Top = 4,
            Width = 5,
            Height = 2,
        });

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
                idTextBox,
                new TextBox(new Rect
                {
                    Left = 10,
                    Top = 3,
                    Width = 10,
                    Height = 1
                }),
                box
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
                    Value = "1234567890"
                },
                new TextBox(Rect.Empty)
                {
                    Value = "abcdef"
                },
                new TextBox(Rect.Empty)
                {
                    Value = "Flash123"
                },
            }
        };


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
                listBox
            }
        };

        _consoleManager.Content = frame;
        _consoleManager.Start();

        Console.WriteLine($"id='{idTextBox.Value}'");

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