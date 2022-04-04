
using SqliteCli.Entities;
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
		case "l":
			ProcessTransList(ss[1]);
			break;
	}

} while (true);

void ProcessTransList(string args)
{
	var req = new ListTransReq();

	if (!ParseDateRange(args, req))
	{
		ParseStartDate(args, req);
	}

	var db = new StockRepo();
	var rc = db.ListTrans2(req);
	foreach (var item in rc)
	{
		Console.WriteLine(item.ToString());
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