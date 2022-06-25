namespace GitCli.Models;

public class CliWindow : ICliWindow
{
    private readonly IServiceProvider _serviceProvider;

    public CliWindow(IServiceProvider serviceProvider)
    {
        this._serviceProvider = serviceProvider;
    }

    public async Task Run(string[] args)
    {
        var commands = new ICommand[]
        {
            _serviceProvider.GetService<GitStatusCommand>()!,
        };
        var cmd = commands.First(x => x.IsMyCommand(args));
        await cmd.Run(args);
    }
}