using AspectCore.Extensions.DependencyInjection;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Prism.Events;
using QueryKits.Extensions;
using QueryKits.Services;
using QueryWeb.Data;
using QueryWeb.Models;
using QueryWeb.Models.Clients;
using QueryWeb.Models.TagHelpers;
using Serilog;


var builder = WebApplication.CreateBuilder(args);

var aspnetcore_env = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT");

builder.Configuration
    .AddJsonFile("appSettings.json", optional: true, reloadOnChange: true)
    .AddJsonFile($"appSettings.{aspnetcore_env}.json", optional: true, reloadOnChange: true);
Log.Logger = new LoggerConfiguration()
    .ReadFrom.Configuration(builder.Configuration)
    .Enrich.FromLogContext()
    //.WriteTo.Console()
    .CreateLogger();
builder.Host.UseSerilog();

Log.Logger.Information($"ENV = '{aspnetcore_env}'");
//builder.WebHost.UseUrls(urls.ToArray());

var connectString = builder.Configuration.GetSection("DbConfig:ConnectionString").Value;
Console.WriteLine($"DB connectString = '{connectString}'");

builder.Services.AddHttpClient();
builder.Services.AddControllers();
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddSingleton<WeatherForecastService>();
builder.Services.AddOptions();
builder.Host.UseServiceProviderFactory(new DynamicProxyServiceProviderFactory());

var pathBaseFeature = new PathBaseFeature
{
    PathBase = "/App1"
};

var configuration = builder.Configuration;
var services = builder.Services;
services.AddSingleton<IQueryEnvironment, QueryEnvironment>();
services.AddEventAggregator(options => options.AutoRefresh = true);
services.AddSingleton<IAppState, AppState>();
services.Configure<DbConfig>(configuration.GetSection("DbConfig"));
services.AddSingleton<IReportRepo, ReportDbContext>();
services.AddTransient<IQueryService, QueryService>();
services.AddTransient<IJsJsonSerializer, JsJsonSerializer>();
services.AddTransient<IJsHelper, JsHelper>();
services.AddTransient<ILanguageService, LanguageService>();
services.AddQueryKits();
services.AddSingleton<IPathBaseFeature>(sp => pathBaseFeature);
services.AddTransient<IPredictNextWordsClient, PredictNextWordsClient>();

var app = builder.Build();
app.UsePathBase(pathBaseFeature.PathBase);

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
}

app.UseStaticFiles();
app.UseRouting();

app.MapBlazorHub();
app.MapControllers();
app.MapFallbackToPage("/_Host");

app.Run();
