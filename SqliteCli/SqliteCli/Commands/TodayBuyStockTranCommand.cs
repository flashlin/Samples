using SqliteCli.Entities;
using SqliteCli.Repos;

namespace SqliteCli.Commands;

public class TodayBuyStockTranCommand : CommandBase
{
    private readonly IStockRepo _stockRepo;

    public TodayBuyStockTranCommand(IStockRepo stockRepo)
    {
        _stockRepo = stockRepo;
    }

    public override bool IsMyCommand(string[] args)
    {
        if (args.Length != 4)
        {
            return false;
        }

        if (args[0] != "b")
        {
            return false;
        }

        return true;
    }

    public override Task Run(string[] args)
    {
        var p = args.ParseArgs<TodayBuyStockTranCommandLine>();
        var tranData = new TransEntity
        {
            TranTime = DateTime.Now.Date,
            StockId = p.StockId,
            StockPrice = p.StockPrice,
            NumberOfShare = p.NumberOfShare
        };
        _stockRepo.BuyStock(tranData);
        return Task.CompletedTask;
    }
}