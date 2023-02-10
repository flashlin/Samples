using SqliteCli.Repos;

namespace SqliteCli.Commands;

public class ListOneStockCommand : CommandBase
{
    private readonly IStockService _stockService;

    public ListOneStockCommand(IStockService stockService)
    {
        _stockService = stockService;
    }

    public override bool IsMyCommand(string[] args)
    {
        if (args.Length != 2)
        {
            return false;
        }

        if (args[0] != "l")
        {
            return false;
        }

        if (DateTime.TryParse(args[1], out _))
        {
            return false;
        }
        return true;
    }

    public override async Task Run(string[] args)
    {
        var commandLine = args.ParseArgs<OneStockCommandArgs>()!;
        var rc = await _stockService.GetOneStockTransAsync(commandLine.StockId);
        rc.DumpList();
    }
}