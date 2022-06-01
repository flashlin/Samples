using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using SqliteCli.Entities;
using SqliteCli.Factories;
using SqliteCli.Helpers;
using SqliteCli.Repos;
using T1.Standard.Extensions;
using StringExtension = SqliteCli.Repos.StringExtension;

namespace SqliteCli;

public class Main
{
    public static string MyAllowSpecificOrigins = "_myAllowSpecificOrigins";
    private IStockService _stockService;
    private IStockRepo _stockRepo;
    private readonly IServiceProvider _serviceProvider;

    public Main(IStockService stockService, IStockRepo stockRepo, 
        IServiceProvider serviceProvider,
        ILogger<Main> logger)
    {
        logger.LogInformation("Startup");
        _stockRepo = stockRepo;
        _serviceProvider = serviceProvider;
        _stockService = stockService;
    }

    public void ShowTransList(string[] args)
    {
        var opts = args.ParseArgs<ShowTransListCommandLineOptions>();
        var rc = _stockService.GetTransList(new ListTransReq
        {
            StartTime = opts?.StartTime,
            EndTime = opts?.EndTime,
            StockId = opts?.StockId,
        });
        rc.Dump();
    }

    public async Task Run(IHost host)
    {
        if (host is WebApplication webApp)
        {
            webApp.UseCors(MyAllowSpecificOrigins);
            webApp.StartAsync(typeof(Program).Assembly, new RuntimeEnvironment());
        }

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
            var commands = new CommandBase[]
            {
                _serviceProvider.GetService<BuyStockTranCommand>()!,
                _serviceProvider.GetService<TodayBuyStockTranCommand>()!,
                _serviceProvider.GetService<QueryStockProfitCommand>()!,
                _serviceProvider.GetService<ListOneStockCommand>()!,
            };
            var cmd = commands.FirstOrDefault(x => x.IsMyCommand(ss));
            if (cmd != null)
            {
                await cmd.Run(ss);
                continue;
            }


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
                    Console.WriteLine(
                        "b 2022/04/04,0056,12.34,1000 :add 2022/04/04 buy stockId:0056 stockPrice:12.34 numberOfShare:1000");
                    Console.WriteLine("d 2022/04/04,1000            :deposit 2022/04/04 amount:1000");
                    Console.WriteLine("append <stockId>             :append stock history data");
                    continue;
                case "l":
                {
                    ShowTransList(ss);
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
                    var opt = ss.ParseArgs<ReportStockCommandLine>()!;
                    if (opt.StockId == null)
                    {
                        await ProcessReportAsync();
                    }
                    else
                    {
                        await ProcessStockReportAsync(opt);
                    }
                
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
                case "show":
                {
                    var stockId = ss[1];
                    await ShowStockHistoryAsync(stockId);
                    break;
                }
                case "t":
                {
                    await _stockService.Test();
                    break;
                }
            }
        } while (true);
    }


    async Task ShowStockHistoryAsync(string stockId)
    {
        await _stockService.ShowStockHistoryAsync(new ShowStockHistoryReq()
        {
            DateRange = new DateRange
            {
                StartDate = DateTime.Now.AddMonths(-12),
                EndDate = DateTime.Now,
            },
            StockId = stockId
        });
    }

    async Task ProcessReportAsync()
    {
        var rc = await _stockService.GetAllStockTransReportAsync();
        rc.Dump();
        _stockService.ShowBalance();
    }

    async Task ProcessStockReportAsync(ReportStockCommandLine commandLine)
    {
        var rc = await _stockService.GetStockReportAsync(new ReportTransReq
        {
            StockId = commandLine.StockId
        });
        rc.DumpList();
        _stockService.ShowBalance();
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

        var db = _stockRepo;
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

        var db = _stockRepo;
        var tranData = new TransEntity
        {
            TranTime = tranDate,
            StockId = stockId,
            StockPrice = stockPrice,
            NumberOfShare = numberOfShare
        };
        db.BuyStock(tranData);
    }
}