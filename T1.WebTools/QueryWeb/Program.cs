using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using QueryKits.Services;
using QueryWeb.Data;

var builder = WebApplication.CreateBuilder(args);

var localEnv = LocalEnvironment.Load();
builder.WebHost.UseUrls($@"http://127.0.0.1:{localEnv.Port}");

builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddSingleton<WeatherForecastService>();

var services = builder.Services;
services.AddSingleton<ILocalEnvironment>(sp => localEnv);
services.AddSingleton<ILocalDbService, LocalDbService>();
services.AddSingleton<IReportRepo, ReportDbContext>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
}


app.UseStaticFiles();

app.UseRouting();

app.MapBlazorHub();
app.MapFallbackToPage("/_Host");

app.Run();
