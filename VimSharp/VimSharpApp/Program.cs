// See https://aka.ms/new-console-template for more information
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using VimSharpApp;
using VimSharpApp.ApiHandlers;

// 主程式
var webApp = CreateWebApplication(args);
var webTask = webApp.RunAsync();

var main = CreateConsoleApplication(args);
main.Run();

// 在主程式結束時觸發 Web 應用程式的關閉
webApp.Lifetime.StopApplication();

// 等待 Web API 完成
await webTask;

// 建立 Web 應用程式的方法
static WebApplication CreateWebApplication(string[] args)
{
    var webBuilder = WebApplication.CreateBuilder(args);

    // 設定日誌，移除 Console 輸出
    webBuilder.Logging.ClearProviders();
    webBuilder.Logging.AddDebug();

    webBuilder.WebHost.UseUrls("http://localhost:8080");
    var webApp = webBuilder.Build();

    // 註冊 API 端點
    JobApiHandler.MapEndpoints(webApp);
    
    return webApp;
}

// 建立 Console 應用程式的方法
static Main CreateConsoleApplication(string[] args)
{
    var builder = Host.CreateApplicationBuilder(args);
    builder.Services.AddSingleton<Main>();
    var host = builder.Build();

    // 取得 Console 應用程式的實例
    return host.Services.GetRequiredService<Main>();
}

