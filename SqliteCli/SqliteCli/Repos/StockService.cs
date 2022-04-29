using CommandLine;
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

    public async Task ShowStockHistoryAsync(StockReportHistoryReq req)
    {
        var tranHistory = _stockRepo.GetStockTranHistory(req);

        var reqDateRange = new DateRange()
        {
            StartDate = req.StartTime,
            EndDate = req.EndTime
        };
        await GetOrUpdateStockHistory(req.StartTime, req.EndTime, req.StockId);

        var stockHistoryReq = ValueHelper.Assign(req, new GetStockHistoryReq());
        stockHistoryReq.StartTime = tranHistory.Select(x => x.TranTime)
            .DefaultIfEmpty(req.StartTime).FirstOrDefault();
        var stockHistory = _stockRepo.GetStockHistory(stockHistoryReq);

        foreach (var month in reqDateRange.GetRangeByMonth())
        {
            var closingDays = stockHistory
                .Count(x => x.TranDate.Year == month.Year && x.TranDate.Month == month.Month);
            var closingSumPrice = stockHistory
                .Where(x => x.TranDate.Year == month.Year && x.TranDate.Month == month.Month)
                .Sum(x => x.ClosingPrice);


            var closingPrice = 0m;
            if (closingDays != 0)
            {
                closingPrice = closingSumPrice / closingDays;
            }

            var valueStr = GetValueStr(closingPrice);
            Console.Write($"{month.ToString("yy-MM")}-");
            Console.BackgroundColor = ConsoleColor.Gray;
            Console.Write($"{valueStr}");
            Console.BackgroundColor = ConsoleColor.Black;
            Console.WriteLine($" {closingPrice.ToNumberString(6)}");
        }
    }

    private string GetValueStr(decimal value)
    {
        var spacesCount = (int) Math.Round(value / 20, MidpointRounding.AwayFromZero);
        return new string(' ', spacesCount);
    }

    public void ShowBalance()
    {
        var balanceInfo = new ReportTranItem
        {
            StockName = "AccountBalance",
            Balance = _stockRepo.GetBalance()
        };
        balanceInfo.DisplayConsoleValue();
    }

    public async Task<List<ReportTranItem>> ReportTransAsync()
    {
        var rc = _stockRepo.ReportTrans(new ReportTransReq());
        foreach (var stock in rc.Where(x => x.TranType == "Buy"))
        {
            await GetOrUpdateStockHistory(stock.MinTranTime, DateTime.Now, stock.StockId);
            var data = _stockRepo.GetStockHistoryData(DateTime.Now.AddDays(-1), stock.StockId);
            stock.CurrentPrice = data.ClosingPrice;
            stock.CurrTotalPrice = data.ClosingPrice * stock.NumberOfShare;
            if (stock.CurrentPrice != 0)
            {
                stock.Profit = stock.Balance + stock.CurrTotalPrice;
            }
            //var totalDays = (int)Math.Round((DateTime.Now - stock.MinTranTime).TotalDays, 0, MidpointRounding.AwayFromZero);
            //stock.InterestRate = stock.Profit / stock.AvgStockPrice * 100 / totalDays;
        }

        return rc;
    }

    private async Task<StockHistoryEntity> GetOrUpdateStockHistory(DateTime startDate, DateTime endDate, string stockId)
    {
        var stockHistory = _stockRepo.GetStockHistoryData(startDate, stockId);
        if (stockHistory == null)
        {
            await AppendStockHistoryRangeFromApi(startDate, endDate, stockId);
            stockHistory = _stockRepo.GetStockHistoryData(startDate, stockId);
            if (stockHistory == null)
            {
                _stockRepo.AppendStockHistory(new StockHistoryEntity()
                {
                    TranDate = startDate.ToDate(),
                    StockId = stockId
                });
            }
        }
        stockHistory = _stockRepo.GetStockHistoryData(startDate, stockId);
        return stockHistory!;
    }

    private async Task AppendStockHistoryRangeFromApi(DateTime startDate, DateTime endDate, string stockId)
    {
        Console.WriteLine($"Query {startDate.ToDateString()} {stockId} from networking...");
        var dataList = await _stockExchangeApi.GetStockTranListAsync(new GetStockReq
        {
            StartDate = startDate,
            EndDate = endDate,
            StockId = stockId,
        });
        foreach (var data in dataList)
        {
            _stockRepo.AppendStockHistory(new StockHistoryEntity
            {
                TranDate = data.Date.ToDate(),
                StockId = stockId,
                TradeVolume = data.TradeVolume,
                DollorVolume = data.DollorVolume,
                TransactionCount = data.Transaction,
                OpeningPrice = data.OpeningPrice,
                ClosingPrice = data.ClosingPrice,
                HighestPrice = data.HighestPrice,
                LowestPrice = data.LowestPrice,
            });
        }
    }

    public void ShowTransList(ShowTransListCommandLineOptions options)
    {
        var rc = _stockRepo.GetTransList(new ListTransReq
        {
            StartTime = options?.StartTime,
            EndTime = options?.EndTime,
            StockId = options?.StockId,
        });
        rc.Dump();
    }

    public async Task Test()
    {
        var resp = await _stockExchangeApi.GetStockTranListAsync(new GetStockReq()
        {
            StockId = "0052",
            StartDate = DateTime.Now.AddDays(-30),
            EndDate = DateTime.Now,
        });
        var list = resp.ToList();
        list.Dump();
    }
}

public class ShowTransListCommandLineOptions
{
    [Value(index: 0, HelpText = "actio name")]
    public string Name { get; set; }

    [Value(index: 1, Required = false, HelpText = "Start tran Date.")]
    public DateTime? StartTime { get; set; }

    [Value(index: 2, Required = false, HelpText = "Start tran Date.")]
    public DateTime? EndTime { get; set; }

    [Option(shortName: 's', Required = false, HelpText = "StockId.")]
    public string? StockId { get; set; }
}