using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using QueryKits.Services;
using QueryWeb.Data;

var builder = WebApplication.CreateBuilder(args);

var localEnv = LocalEnvironment.Load();
var urls = new List<string>
{
    $@"http://0.0.0.0:{localEnv.Port}"
};
if (!LocalEnvironment.IsPortUsed(80))
{
    urls.Add("http://0.0.0.0:80");
}
builder.WebHost.UseUrls(urls.ToArray());

builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddSingleton<WeatherForecastService>();

var services = builder.Services;
services.AddSingleton<ILocalEnvironment>(sp => localEnv);
//services.AddSingleton<ILocalDbService, LocalDbService>();
//services.AddSingleton<IReportRepo, ReportDbContext>();

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
