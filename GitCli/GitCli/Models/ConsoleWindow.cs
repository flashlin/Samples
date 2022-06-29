using T1.Standard.Extensions;

namespace GitCli.Models;

public class ConsoleWindow : IConsoleWindow
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ConsoleWriter _console;

    public ConsoleWindow(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
        _console = new ConsoleWriter();
    }

    public async Task Run(string[] args)
    {
        do
        {
            var lineCommand = Console.ReadLine();
            var lineArgs = lineCommand.ParseCommandArgsLine().ToArray();

            var commands = new ICommand[]
            {
                _serviceProvider.GetService<GitStatusCommand>()!,
            };
            var cmd = commands.FirstOrDefault(x => x.IsMyCommand(lineArgs));
            if (cmd == null)
            {
                _console.SetForegroundColor(ConsoleColor.Red);
                _console.WriteLine($"Unknown command: {lineCommand}");
                _console.ResetColor();
                continue;
            }
            await cmd.Run();
        } while (true);
    }
}