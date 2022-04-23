using SqliteCli.Entities;

namespace SqliteCli.Repos;

public class StockService : IStockService
{
    private readonly IStockRepo _stockRepo;
    private readonly IStockExchangeApi _stockExchangeApi;

    public StockService(IStockRepo stockRepo, IStockExchangeApi stockExchangeApi)
    {
        _stockRepo = stockRepo;
        _stockExchangeApi = stockExchangeApi;
    }
        
    public async Task<List<ReportTranItem>> ReportTransAsync()
    {
        var rc = _stockRepo.ReportTrans(new ReportTransReq());
        var api = _stockExchangeApi;
        foreach (var stock in rc.Where(x => x.TranType == "Buy"))
        {
            var data = await api.GetLastDataAsync(stock.StockId);
            stock.CurrentPrice = data.ClosingPrice;
            stock.CurrTotalPrice = data.ClosingPrice * stock.NumberOfShare;
            if (stock.CurrentPrice != 0)
            {
                stock.Profit = stock.Balance + stock.CurrTotalPrice;
            }

            _stockRepo.UpsertStockHistory(new StockHistoryEntity
            {
                TranDate = data.Date,
                StockId = stock.StockId,
                TradeVolume = data.TradeVolume,
                DollorVolume = data.DollorVolume,
                TransactionCount = data.Transaction,
                OpeningPrice = data.OpeningPrice,
                ClosingPrice = data.ClosingPrice,
                HighestPrice = data.HighestPrice,
                LowestPrice = data.LowestPrice,
            });

            //var totalDays = (int)Math.Round((DateTime.Now - stock.MinTranTime).TotalDays, 0, MidpointRounding.AwayFromZero);
            //stock.InterestRate = stock.Profit / stock.AvgStockPrice * 100 / totalDays;
        }

        return rc;
    }
}