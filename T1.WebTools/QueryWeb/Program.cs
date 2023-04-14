using AspectCore.Extensions.DependencyInjection;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Prism.Events;
using QueryKits.Extensions;
using QueryKits.Services;
using QueryWeb.Data;
using QueryWeb.Models;
using Serilog;


var builder = WebApplication.CreateBuilder(args);

var aspnetcore_env = Environment.GetEnvironmentVariable("DOTNET_ENVIRONMENT");
Console.WriteLine($"ENV = '{aspnetcore_env}'");

builder.Configuration
    .AddJsonFile("appSettings.json", optional: true, reloadOnChange: true)
    .AddJsonFile($"appSettings.{aspnetcore_env}.json", optional: true, reloadOnChange: true);
Log.Logger = new LoggerConfiguration()
    .ReadFrom.Configuration(builder.Configuration)
    .Enrich.FromLogContext()
    //.WriteTo.Console()
    .CreateLogger();
builder.Host.UseSerilog();

var localEnv = LocalEnvironment.Load();
//var urls = new List<string>();
//if (!LocalEnvironment.IsPortUsed(80))
//{
//    urls.Add("http://0.0.0.0:80");
//}
//else
//{
//    urls.Add($@"http://127.0.0.1:{localEnv.Port}");
//}
//builder.WebHost.UseUrls(urls.ToArray());

var connectString = builder.Configuration.GetSection("DbConfig:ConnectionString").Value;
Console.WriteLine($"DB connectString = '{connectString}'");

builder.Services.AddControllers();
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddSingleton<WeatherForecastService>();
builder.Services.AddOptions();
builder.Host.UseServiceProviderFactory(new DynamicProxyServiceProviderFactory());

var configuration = builder.Configuration;
var services = builder.Services;
services.AddSingleton<IQueryEnvironment, QueryEnvironment>();
services.AddSingleton<IEventAggregator, EventAggregator>();
services.AddSingleton<ILocalEnvironment>(sp => localEnv);
services.AddSingleton<IAppState, AppState>();
//services.AddSingleton<ILocalDbService, LocalDbService>();
services.Configure<DbConfig>(configuration.GetSection("DbConfig"));
services.AddSingleton<IReportRepo, ReportDbContext>();
services.AddTransient<IQueryService, QueryService>();
services.AddTransient<IJsJsonSerializer, JsJsonSerializer>();
services.AddTransient<IJsHelper, JsHelper>();
services.AddTransient<ILanguageService, LanguageService>();
services.AddQueryKits();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
}

app.UseStaticFiles();
app.UseRouting();

app.MapBlazorHub();
app.UseEndpoints(endpoints=>
{
    endpoints.MapControllers();
});
app.MapFallbackToPage("/_Host");

app.Run();
