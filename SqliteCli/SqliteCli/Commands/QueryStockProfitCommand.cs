using SqliteCli.Repos;

namespace SqliteCli.Commands;

public class QueryStockProfitCommand : CommandBase
{
    private readonly IStockService _stockService;
    private IStockRepo _stockRepo;
    public QueryStockProfitCommand(IStockService stockService, IStockRepo stockRepo)
    {
        _stockService = stockService;
        _stockRepo = stockRepo;
    }
    
    public override bool IsMyCommand(string[] args)
    {
        if (args.Length != 1)
        {
            return false;
        }

        if (args[0] != "rr")
        {
            return false;
        }

        return true;
    }

    public override async Task Run(string[] args)
    {
        var p = args.ParseArgs<QueryStockProfitCommandLine>()!;
        var rc = await _stockService.GetAllStockProfitReportAsync();
        rc.DumpList();
    }
}