using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Prism.Events;
using QueryKits.Services;
using QueryWeb.Data;
using QueryWeb.Models;
using Serilog;


var builder = WebApplication.CreateBuilder(args);

builder.Configuration
    .AddJsonFile("appSettings.json", optional: true, reloadOnChange: true);
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

builder.Services.AddControllers();
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddSingleton<WeatherForecastService>();
builder.Services.AddOptions();

var configuration = builder.Configuration;
var services = builder.Services;
services.AddSingleton<IEventAggregator, EventAggregator>();
services.AddSingleton<ILocalEnvironment>(sp => localEnv);
services.AddSingleton<IAppState, AppState>();
//services.AddSingleton<ILocalDbService, LocalDbService>();
services.Configure<DbConfig>(configuration.GetSection("DbConfig"));
services.AddSingleton<IReportRepo, ReportDbContext>();
services.AddTransient<IQueryService, QueryService>();
services.AddTransient<IJsJsonSerializer, JsJsonSerializer>();
services.AddTransient<IJsHelper, JsHelper>();

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
