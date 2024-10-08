using SqliteCli.Entities;
using SqliteCli.Repos;

namespace SqliteCli.Commands;

public class BuyStockTranCommand : CommandBase
{
    private IStockRepo _stockRepo;
    public BuyStockTranCommand(IStockRepo stockRepo)
    {
        _stockRepo = stockRepo;
    }
    
    public override bool IsMyCommand(string[] args)
    {
        if (args.Length != 5)
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
        var p = args.ParseArgs<BuyStockTranCommandLine>()!;
        var tranData = new TransEntity
        {
            TranTime = p.TranDate,
            StockId = p.StockId,
            StockPrice = p.StockPrice,
            NumberOfShare = p.NumberOfShare
        };
        _stockRepo.BuyStock(tranData);
        return Task.CompletedTask;
    }
}