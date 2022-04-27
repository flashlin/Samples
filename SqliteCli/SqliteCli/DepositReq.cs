
using CommandLine;
using SqliteCli.Repos;

public class DepositReq
{
	public DateTime TranTime { get; set; }
	public decimal Balance { get; set; }
}

public class ConsoleApp
{
    private IStockService _stockService;

    public ConsoleApp(IStockService stockService)
    {
        _stockService = stockService;
    }
    
    public void ShowTransList(string[] args)
    {
        var opts = Parser.Default.ParseArguments<ShowTransListCommandLineOptions>(args)
            .MapResult((opts) => { return opts; },
                errs => { return null; });
        _stockService.ShowTransList(opts);
    }
}