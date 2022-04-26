using CsvCli.Repositories;
using CsvCli.Services;
using Microsoft.Extensions.DependencyInjection;

var services = new ServiceCollection();
//services.AddHttpClient();
//services.AddHttpClient<IWebApiClient, WebApiClient>();
services.AddDbContext<CsvDbContext>();
services.AddSingleton<ConsoleApp>();
var sp = services.BuildServiceProvider();

var app = sp.GetService<ConsoleApp>();
app!.Run(args);
