using Microsoft.Extensions.DependencyInjection;
using SqliteCli.Entities;
using SqliteCli.Repos;
using System.Text.RegularExpressions;
using Serilog;
using SqliteCli;
using SqliteCli.Helpers;
using T1.Standard.Collections.Generics;
using T1.Standard.Extensions;
using T1.Standard.Web;
using Serilog.Events;
using Microsoft.Extensions.Hosting;
using SqliteCli.Factories;

Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Information()
    .MinimumLevel.Override("Microsoft.AspNetCore", LogEventLevel.Warning)
    .Enrich.FromLogContext()
    .WriteTo.File("d:/demo/logs/myapp.txt", rollingInterval: RollingInterval.Day)
    .CreateLogger();

try
{
    var host = new HostFactory().Create(args)
        .ConfigureServices(services =>
        {
            services.AddControllers();
            services.AddHttpClient();
            services.AddHttpClient<IWebApiClient, WebApiClient>();
            //services.AddTransient<IStockExchangeApi, TwseStockExchangeApi>();
            services.AddTransient<IStockExchangeApi, FinMindApi>();
            services.AddDbContext<StockDbContext>();
            services.AddTransient<IStockRepo, StockRepo>();
            services.AddTransient<IStockService, StockService>();
            services.AddSingleton<Startup>();
        })
        .UseSerilog()
        .Build();

    var app = host.Services.GetService<Startup>();
    await app!.Run(host);
}
catch (Exception ex)
{
    Log.Fatal(ex, "Application Fail");
}
finally
{
    Log.CloseAndFlush();
}