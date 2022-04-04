
using SqliteCli.Entities;
using SqliteCli.Repos;
using System.Text.RegularExpressions;
using T1.Standard.Extensions;


do
{
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
			Console.WriteLine("l                       :list trans history");
			Console.WriteLine("l 2022/04/04            :list trans history from 2022/04/04");
			Console.WriteLine("l 2022/04/04-2022-04-05 :list trans history from 2022/04/04~2022/04/05");
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
	}

} while (true);

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
	if(!int.TryParse(numberOfShareStr, out var numberOfShare))
	{
		Console.WriteLine($"Parse '{numberOfShareStr}' fail, please input 2022/04/04,0056,33.1,<1000>");
		return;
	}

	var db = new StockRepo();
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

	var db = new StockRepo();
	var rc = db.ListTrans(req);
	if (rc.Count > 0)
	{
		var title = rc.First().GetDisplayTitle();
		Console.WriteLine(title);
	}
	foreach (var item in rc)
	{
		Console.WriteLine(item.GetDisplayValue());
	}

	if (rc.Count > 0)
	{
		var summary = new TransHistory
		{
			TranTime = DateTime.Now,
			TranType = "Summary",
			NumberOfShare = rc.Sum(x => x.NumberOfShare),
			HandlingFee = rc.Sum(x => x.HandlingFee),
			Balance = rc.Sum(x => x.Balance),
		};
		Console.WriteLine(summary.GetDisplayValue());
	}
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