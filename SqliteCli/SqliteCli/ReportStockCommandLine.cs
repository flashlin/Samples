using CommandLine;
using SqliteCli.Repos;
using T1.Standard.Extensions;

namespace SqliteCli;

public class ReportStockCommandLine
{
    [Value(index: 0, HelpText = "action name")]
    public string ActionName { get; set; } = "";

    [Value(index: 1, Required = false, HelpText = "StockId")]
    public string? StockId { get; set; }
}


public class OneStockCommandLine
{
    [Value(index: 0, HelpText = "action name")]
    public string ActionName { get; set; } = "l";

    [Value(index: 1, Required = true, HelpText = "StockId")]
    public string StockId { get; set; } = String.Empty;
}

public class ListOneStockCommand : CommandBase
{
    private IStockService _stockService;

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
        var commandLine = args.ParseArgs<OneStockCommandLine>()!;
        var rc = await _stockService.GetOneStockTransAsync(commandLine.StockId);
        rc.DumpList();
    }
}

public class AddDiscountTranCommand
{
    [Value(index: 0, HelpText = "action name")]
    public string ActionName { get; set; } = "";
    
    [Value(index: 1, Required = false, HelpText = "TranDate")]
    public DateTime? TranDate { get; set; }
    
    [Value(index: 2, Required = false, HelpText = "amount")]
    public decimal Amount { get; set; }

    [Value(index: 3, Required = false, HelpText = "remark")]
    public string Remark { get; set; } = "";
}
