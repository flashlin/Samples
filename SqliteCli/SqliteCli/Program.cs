
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
services.AddSingleton<ConsoleApp>();

var serviceProvider = services.BuildServiceProvider();

var app = serviceProvider.GetService<ConsoleApp>();
await app!.Run();
