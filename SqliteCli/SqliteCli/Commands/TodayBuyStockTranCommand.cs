using SqliteCli.Entities;
using SqliteCli.Repos;

namespace SqliteCli.Commands;

public class DividendCommand : CommandBase
{
    private readonly IStockRepo _stockRepo;

    public DividendCommand(IStockRepo stockRepo)
    {
        _stockRepo = stockRepo;
    }
    
    public override bool IsMyCommand(string[] args)
    {
        if (args[0] != "di")
        {
            return false;
        }
        return true;
    }

    public override Task Run(string[] args)
    {
        var p = args.ParseArgs<DividendCommandLine>()!;
        var tranData = new TransEntity
        {
            TranTime = p.TranTime.Date,
            TranType = "Dividend",
            StockId = p.StockId,
            NumberOfShare = p.NumberOfShare,
            StockPrice = 0,
            Balance = p.Dividend,
        };
        _stockRepo.AddTrans(tranData);
        return Task.CompletedTask;
    }
}

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