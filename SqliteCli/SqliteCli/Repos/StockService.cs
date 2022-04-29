using CommandLine;
using SqliteCli.Entities;
using SqliteCli.Helpers;
using T1.Standard.Common;
using T1.Standard.Extensions;

namespace SqliteCli.Repos;

public class ShowStockHistoryReq
{
	public DateTime StartTime { get; set; }
	public DateTime EndTime { get; set; }
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

	public StockService(IStockRepo stockRepo, IStockExchangeApi stockExchangeApi)
	{
		_stockRepo = stockRepo;
		_stockExchangeApi = stockExchangeApi;
	}

	public async Task ShowStockHistoryAsync(ShowStockHistoryReq req)
	{
		var tranHistory = _stockRepo.GetStockTranHistory(req);

		var reqDateRange = req.DateRange;
		await EnsuredStockHistory(req.DateRange, req.StockId);

		var stockHistoryReq = new GetStockHistoryReq()
		{
			StartTime = tranHistory.Select(x => x.TranTime)
				.DefaultIfEmpty(req.StartTime).FirstOrDefault(),
			EndTime = req.EndTime,
			StockId = req.StockId
		};
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
		var spacesCount = (int)Math.Round(value / 20, MidpointRounding.AwayFromZero);
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
			await EnsuredStockHistory(new DateRange()
			{
				StartDate = stock.MinTranTime,
				EndDate = DateTime.Now,
			}, stock.StockId);
			
			var data = _stockRepo.GetLastStockHistoryData(stock.StockId);
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
		var resp = await _stockExchangeApi.GetStockHistoryListAsync(new GetStockReq()
		{
			DateRange = new DateRange() {
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