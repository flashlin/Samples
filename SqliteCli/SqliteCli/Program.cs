
using Microsoft.Extensions.DependencyInjection;
using SqliteCli.Entities;
using SqliteCli.Repos;
using System.Text.RegularExpressions;
using SqliteCli.Helpers;
using T1.Standard.Collections.Generics;
using T1.Standard.Extensions;
using T1.Standard.Web;

var services = new ServiceCollection();
services.AddHttpClient();
services.AddHttpClient<IWebApiClient, WebApiClient>();
services.AddTransient<IStockExchangeApi, TwseStockExchangeApi>();
services.AddDbContext<StockDbContext>();
services.AddTransient<IStockRepo, StockRepo>();
services.AddTransient<IStockService, StockService>();

var serviceProvider = services.BuildServiceProvider();

do
{
	Console.WriteLine();
	Console.Write("$ ");
	var commandLine = Console.ReadLine();
	if (string.IsNullOrEmpty(commandLine))
	{
		Console.WriteLine("type '?' to get help");
		continue;
	}

	var ss = commandLine.Split(' ');
	var command = ss[0];
	switch (command)
	{
		case "q":
			Console.WriteLine("END");
			return;
		case "?":
			Console.WriteLine("l                            :list trans history");
			Console.WriteLine("l 2022/04/04                 :list trans history from 2022/04/04");
			Console.WriteLine("l 2022/04/04-2022-04-05      :list trans history from 2022/04/04~2022/04/05");
			Console.WriteLine("b 2022/04/04,0056,12.34,1000 :add 2022/04/04 buy stockId:0056 stockPrice:12.34 numberOfShare:1000");
			Console.WriteLine("d 2022/04/04,1000            :deposit 2022/04/04 amount:1000");
			Console.WriteLine("append <stockId>             :append stock history data");
			continue;
		case "l":
			{
				var cmdArgs = string.Empty;
				if (ss.Length > 1)
				{
					cmdArgs = ss[1];
				}
				ProcessTransList(cmdArgs);
				break;
			}
		case "b":
			{
				var cmdArgs = string.Empty;
				if (ss.Length > 1)
				{
					cmdArgs = ss[1];
				}
				ProcessBuyStock(cmdArgs);
				break;
			}
		case "r":
			{
				var cmdArgs = string.Empty;
				if (ss.Length > 1)
				{
					cmdArgs = ss[1];
				}
				await ProcessReportAsync(cmdArgs);
				break;
			}
		case "d":
			{
				var cmdArgs = string.Empty;
				if (ss.Length > 1)
				{
					cmdArgs = ss[1];
				}
				ProcessDeposit(cmdArgs);
				break;
			}
		case "append":
		{
			var stockId = ss[1];
			await AppendStockHistoryAsync(stockId);
			break;
		}
		case "show":
		{
			var stockId = ss[1];
			ShowStockHistoryAsync(stockId);
			break;
		}
	}

} while (true);

void ShowStockHistoryAsync(string stockId)
{
	var stockService = serviceProvider.GetService<IStockService>();
	 stockService.ShowStockHistory(new StockReportHistoryReq()
	{
		StartTime = DateTime.Parse("2021-01-01"),
		EndTime = DateTime.Now,
		StockId = stockId
	});
}

async Task AppendStockHistoryAsync(string stockId)
{
	var stockService = serviceProvider.GetService<IStockService>();
	await stockService.AppendStockHistoryAsync(new DateRange()
	{
		StartDate = DateTime.Parse("2021/01/01"),
		EndDate = DateTime.Today
	}, stockId);
}
	

async Task ProcessReportAsync(string cmdArgs)
{
	var stockService = serviceProvider.GetService<IStockService>();
	var rc = await stockService.ReportTransAsync();
	rc.Dump();
	stockService.ShowBalance();
}

void ProcessDeposit(string dataText)
{
	var tranDatePattern = RegexPattern.Group("tranDate", @"\d{4}/\d{2}/\d{2}");
	var amountPattern = RegexPattern.Group("amount", @"\d+");
	var rg = new Regex($"{tranDatePattern},{amountPattern}");
	var m = rg.Match(dataText);
	if (!m.Success)
	{
		Console.WriteLine("parse fail, please input 2022/04/04,12345");
		return;
	}

	var tranDateStr = m.Groups["tranDate"].Value;
	if (!DateTime.TryParse(tranDateStr, out var tranDate))
	{
		Console.WriteLine($"parse '{tranDateStr}' fail, example: 2022/04/04");
		return;
	}

	var amountStr = m.Groups["amount"].Value;
	if (!decimal.TryParse(amountStr, out var amount))
	{
		Console.WriteLine($"Parse '{amountStr}' fail, example: 12345");
		return;
	}

	var db = serviceProvider.GetService<IStockRepo>();
	db.Deposit(new DepositReq
	{
		TranTime = tranDate,
		Balance = amount
	});
}

//data = "2022/04/04,0050,10.0,1000"
void ProcessBuyStock(string dataText)
{
	var tranDatePattern = RegexPattern.Group("tranDate", @"\d{4}/\d{2}/\d{2}");
	var stockIdPattern = RegexPattern.Group("stockId", @"[^,]+");
	var stockPricePattern = RegexPattern.Group("stockPrice", @"[^,]+");
	var numberOfSharePattern = RegexPattern.Group("numberOfShare", @"[^,]+");
	var rg = new Regex($"{tranDatePattern},{stockIdPattern},{stockPricePattern},{numberOfSharePattern}");
	var m = rg.Match(dataText);
	if (!m.Success)
	{
		Console.WriteLine("parse fail, please input 2022/04/04,0056,33.1,1000");
		return;
	}

	var tranDateStr = m.Groups["tranDate"].Value;
	if (!DateTime.TryParse(tranDateStr, out var tranDate))
	{
		Console.WriteLine($"parse '{tranDateStr}' fail, please input <2022/04/04>,0056,33.1,1000");
		return;
	}

	var stockId = m.Groups["stockId"].Value;

	var stockPriceStr = m.Groups["stockPrice"].Value;
	if (!decimal.TryParse(stockPriceStr, out var stockPrice))
	{
		Console.WriteLine($"Parse '{stockPriceStr}' fail, please input 2022/04/04,0056,<33.1>,1000");
		return;
	}

	var numberOfShareStr = m.Groups["numberOfShare"].Value;
	if (!int.TryParse(numberOfShareStr, out var numberOfShare))
	{
		Console.WriteLine($"Parse '{numberOfShareStr}' fail, please input 2022/04/04,0056,33.1,<1000>");
		return;
	}

	var db = serviceProvider.GetService<IStockRepo>();
	var tranData = new TransEntity
	{
		TranTime = tranDate,
		StockId = stockId,
		StockPrice = stockPrice,
		NumberOfShare = numberOfShare
	};
	db.BuyStock(tranData);
}


void ProcessTransList(string dateRange)
{
	var req = new ListTransReq();

	if (!ParseDateRange(dateRange, req))
	{
		ParseStartDate(dateRange, req);
	}

	var db = serviceProvider.GetService<IStockRepo>();
	var rc = db.ListTrans(req);
	rc.Dump();
}

bool ParseDateRange(string args, ListTransReq req)
{
	var startTime = RegexPattern.Group("startTime", @"\d{4}/\d{2}/\d{2}");
	var endTime = RegexPattern.Group("endTime", @"\d{4}/\d{2}/\d{2}");
	var dateRange = @$"{startTime}\-{endTime}";
	var rg = new Regex(dateRange);

	var m = rg.Match(args);
	if (m.Success)
	{
		req.StartTime = DateTime.Parse(m.Groups["startTime"].Value);
		req.EndTime = DateTime.Parse(m.Groups["endTime"].Value);
		return true;
	}

	return false;
}

bool ParseStartDate(string args, ListTransReq req)
{
	var startTime = RegexPattern.Group("startTime", @"\d{4}/\d{2}/\d{2}");
	var startDateRg = new Regex(@$"{startTime}\-");
	var m = startDateRg.Match(args);
	if (m.Success)
	{
		req.StartTime = DateTime.Parse(m.Groups["startTime"].Value);
		return true;
	}


	var startDateRg2 = new Regex(@$"{startTime}");
	var m2 = startDateRg2.Match(args);
	if (m2.Success)
	{
		req.StartTime = DateTime.Parse(m2.Groups["startTime"].Value);
		return true;
	}
	return false;
}