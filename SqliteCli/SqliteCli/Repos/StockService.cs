using SqliteCli.Entities;
using SqliteCli.Helpers;
using T1.Standard.Common;

namespace SqliteCli.Repos;

public class StockReportHistoryReq
{
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public string StockId { get; set; }
}

public class StockReportHistory
{
    public class Item
    {
        public decimal Value { get; set; }
        public int Month { get; set; }
        public decimal YValue { get; set; }
    }

    public Dictionary<DateTime, Item> Items { get; set; }
}

public class StockService : IStockService
{
    private readonly IStockRepo _stockRepo;
    private readonly IStockExchangeApi _stockExchangeApi;

    public StockService(IStockRepo stockRepo, IStockExchangeApi stockExchangeApi)
    {
        _stockRepo = stockRepo;
        _stockExchangeApi = stockExchangeApi;
    }

    public async Task AppendStockHistoryAsync(DateRange dateRange, string stockId)
    {
        foreach (var date in dateRange.GetRangeByDay())
        {
            var stockData = _stockRepo.GetStockHistoryData(date, stockId);
            if (stockData == null)
            {
                Console.WriteLine($"query {date.ToString("yyyy/MM/dd")} {stockId}");
                var resp = await _stockExchangeApi.GetStockTranListAsync(new GetStockReq()
                {
                    Date = date,
                    StockId = stockId
                });
                var stockNetworkData = resp.ToArray();
                foreach (var stockDateData in stockNetworkData)
                {
                    var data = ValueHelper.Assign(stockDateData, new StockHistoryEntity());
                    data.TranDate = stockDateData.Date;
                    data.TransactionCount = stockDateData.Transaction;
                    _stockRepo.AppendStockHistory(data);
                }
            }
        }
        Console.WriteLine("done");
    }

    public void ShowStockHistory(StockReportHistoryReq req)
    {
        var stockHistory = _stockRepo.GetStockHistory(ValueHelper.Assign(req, new GetStockHistoryReq()));
        //var tranHistory = _stockRepo.GetStockTranHistory(req);

        var dateRange = new DateRange()
        {
            StartDate = req.StartTime,
            EndDate = req.EndTime
        };
        foreach (var date in dateRange.GetRangeByMonth())
        {
            var item = stockHistory
                .Where(x => x.TranDate == date)
                .DefaultIfEmpty(new StockHistoryEntity
                {
                    TranDate = date,
                }).FirstOrDefault();

            var valueStr = GetValueStr(item.ClosingPrice);
            Console.Write($"{date.ToString("yy-MM")}-");
            Console.BackgroundColor = ConsoleColor.Gray;
            Console.Write($"{valueStr}");
            Console.BackgroundColor = ConsoleColor.Black;
            Console.WriteLine($" {item.ClosingPrice.ToNumberString(6)}");
        }
    }

    private string GetValueStr(decimal value)
    {
        var spacesCount = (int)Math.Round(value / 20, MidpointRounding.AwayFromZero);
        return new string(' ', spacesCount);
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

            _stockRepo.AppendStockHistory(new StockHistoryEntity
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