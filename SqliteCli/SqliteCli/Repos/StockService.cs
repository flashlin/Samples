using CommandLine;
using SqliteCli.Entities;
using SqliteCli.Helpers;
using T1.Standard.Common;
using T1.Standard.Extensions;

namespace SqliteCli.Repos;

public class ShowStockHistoryReq
{
    public string StockId { get; set; }
    public DateRange DateRange { get; set; }
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
    private readonly StockDbContext _db;

    public StockService(IStockRepo stockRepo, IStockExchangeApi stockExchangeApi,
        StockDbContext db)
    {
        _stockRepo = stockRepo;
        _stockExchangeApi = stockExchangeApi;
        _db = db;
    }

    public async Task ShowStockHistoryAsync(ShowStockHistoryReq req)
    {
        var tranHistory = _stockRepo.GetStockTranHistory(req);

        await EnsuredStockHistory(req.DateRange, req.StockId);

        var stockHistoryReq = new GetStockHistoryReq()
        {
            StartTime = req.DateRange.StartDate.StartOfMonth(),
            EndTime = req.DateRange.EndDate,
            StockId = req.StockId
        };

        var stockHistory = _stockRepo.GetStockHistory(stockHistoryReq);
        foreach (var month in req.DateRange.GetRangeByMonth())
        {
            var closingDays = stockHistory
                .Count(x => x.TranDate.EqualYearMonth(month));
            var closingSumPrice = stockHistory
                .Where(x => x.TranDate.EqualYearMonth(month))
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
        var spacesCount = (int) Math.Round(value / 10, MidpointRounding.AwayFromZero);
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

    public async Task<List<ReportTranItem>> GetAllStockTransReportAsync()
    {
        var rc = _stockRepo.GetTransGroupByStock(new ReportTransReq());
        foreach (var stock in rc.Where(x => x.TranType == "Buy"))
        {
            await EnsuredStockHistory(new DateRange()
            {
                StartDate = stock.MinTranTime,
                EndDate = DateTime.Now,
            }, stock.StockId);

            var data = _stockRepo.GetLastStockHistoryData(stock.StockId)!;
            stock.CurrentPrice = data.ClosingPrice;
            stock.CurrTotalPrice = data.ClosingPrice * stock.NumberOfShare;
            if (stock.CurrentPrice != 0)
            {
                stock.Profit = stock.Balance + stock.CurrTotalPrice;
            }
        }
        
        foreach (var stock in rc.Where(x => x.TranType == "Dividend").ToArray())
        {
            var buyStock = rc.First(x => x.StockId == stock.StockId && x.TranType == "Buy");
            stock.Profit = stock.Balance + buyStock.Profit;
            var idx = rc.IndexOf(stock);
            rc.Insert(idx+1, new ReportTranItem
            {
                StockId = buyStock.StockId,
                StockName = "",
                Profit = -stock.Profit + stock.Balance,
            });
        }
        
        foreach (var stock in rc.Where(x => x.TranType == "Sale"))
        {
            await EnsuredStockHistory(new DateRange()
            {
                StartDate = stock.MinTranTime,
                EndDate = DateTime.Now,
            }, stock.StockId);

            stock.Profit = stock.Balance;
        }

        return rc;
    }
    
    public async Task<List<ReportProfitItem>> GetAllStockProfitReportAsync()
    {
        var result = new List<ReportProfitItem>();
        
        var rc = _stockRepo.GetTransGroupByStock(new ReportTransReq());
        foreach (var stock in rc.Where(x => x.TranType == "Buy"))
        {
            await EnsuredStockHistory(new DateRange()
            {
                StartDate = stock.MinTranTime,
                EndDate = DateTime.Now,
            }, stock.StockId);

            var dateRange = new DateRange(DateTime.Now.AddDays(-5), DateTime.Now);
            foreach (var date in dateRange.GetRangeByDay())
            {
                var reportProfitItem = QueryProfit(stock, date);
                result.Add(reportProfitItem);
            }
        }
        
        return result;
    }

    private ReportProfitItem QueryProfit(ReportTranItem stock, DateTime queryDate)
    {
        var data = _db.StocksHistory
            .Where(x => x.StockId == stock.StockId && x.TranDate == queryDate.ToDate())
            .Select(x => new {x.StockId, x.ClosingPrice})
            .First();

        var currentPrice = data.ClosingPrice;
        var currentTotal = data.ClosingPrice * stock.NumberOfShare;
        var profit = 0m;
        if (currentTotal != 0)
        {
            profit = stock.Balance + currentTotal;
        }

        var reportProfitItem = new ReportProfitItem()
        {
            StockId = stock.StockId,
            StockName = stock.StockName,
            Date = queryDate, 
            StockPrice = currentPrice,
            Profit = profit
        };
        return reportProfitItem;
    }

    public async Task<List<TransHistory>> GetOneStockTransAsync(string stockId)
    {
        return _stockRepo.GetOneStockTransList(stockId);
    }

    public async Task<List<ReportTranItem>> GetStockReportAsync(ReportTransReq req)
    {
        var stock = _stockRepo
            .GetTransGroupByStock(req)
            .FirstOrDefault(x => x.TranType == "Buy");
        
        if (stock == null)
        {
            return new List<ReportTranItem>();
        }
        
        await EnsuredStockHistory(new DateRange()
        {
            StartDate = stock.MinTranTime,
            EndDate = DateTime.Now,
        }, stock.StockId);

        var report = new List<ReportTranItem>();
        foreach (var day in new DateRange(stock.MinTranTime, DateTime.Now).GetRangeByDay())
        {
            var data = _stockRepo.GetStockHistoryData(day, stock.StockId)!;
            if (data.ClosingPrice == 0)
            {
                continue;
            }
            var stockDay = ValueHelper.Assign(stock, new ReportTranItem());
            stockDay.MinTranTime = day;
            stockDay.CurrentPrice = data.ClosingPrice;
            stockDay.CurrTotalPrice = data.ClosingPrice * stock.NumberOfShare;
            if (stockDay.CurrentPrice != 0)
            {
                stockDay.Profit = stock.Balance + stockDay.CurrTotalPrice;
            }

            if (stockDay.TranType == "Sale")
            {
                stockDay.Profit = stock.Balance;
            }
            report.Add(stockDay);
        }
        
        return report;
    }


    private async Task EnsuredStockHistory(DateRange dateRange, string stockId)
    {
        var first = true;
        foreach (var date in dateRange.GetRangeByDay())
        {
            var stockHistory = _stockRepo.GetStockHistoryData(date, stockId);
            if (stockHistory == null && date.IsNowClosingTime())
            {
                if (first)
                {
                    Console.WriteLine($"Beacuse {date} {stockId}");
                    await AppendStockHistoryRangeFromApi(dateRange, stockId);
                }
                else
                {
                    Console.WriteLine($"ERROR {date.ToDateString()} {stockId}");
                    throw new Exception();
                }

                first = false;
                stockHistory = _stockRepo.GetStockHistoryData(date, stockId);
                if (stockHistory == null)
                {
                    _stockRepo.AppendStockHistory(new StockHistoryEntity()
                    {
                        TranDate = date.ToDate(),
                        StockId = stockId
                    });
                }
            }
        }
    }

    public async Task AppendStockHistoryRangeFromApi(DateRange dateRange, string stockId)
    {
        Console.WriteLine($"Query {dateRange} {stockId} from networking...");
        var dataList = await _stockExchangeApi.GetStockHistoryListAsync(new GetStockReq
        {
            DateRange = dateRange,
            StockId = stockId,
        }).ToListAsync();
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

        foreach (var date in dateRange.GetRangeByDay())
        {
            if (!IsCanAppend(date))
            {
                continue;
            }

            _stockRepo.AppendStockHistory(new StockHistoryEntity
            {
                TranDate = date.ToDate(),
                StockId = stockId,
            });
        }
    }

    private static bool IsCanAppend(DateTime date)
    {
        return date.IsWorkDay() && DateTime.Now.ToDate() != date.ToDate();
    }

    public List<TransHistory> GetTransList(ListTransReq listTransReq)
    {
        var rc = _stockRepo.GetTransList(listTransReq);
        return rc;
    }

    public async Task Test()
    {
        var resp = await _stockExchangeApi.GetStockHistoryListAsync(new GetStockReq()
        {
            DateRange = new DateRange()
            {
                StartDate = DateTime.Now.AddDays(-30),
                EndDate = DateTime.Now,
            },
            StockId = "0052",
        }).ToListAsync();
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