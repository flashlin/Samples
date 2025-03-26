// See https://aka.ms/new-console-template for more information
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using VimSharpApp;
using VimSharpApp.ApiHandlers;

// 建立 Web 應用程式
var webBuilder = WebApplication.CreateBuilder(args);
webBuilder.WebHost.UseUrls("http://localhost:8080");
var webApp = webBuilder.Build();

// 註冊 API 端點
webApp.MapWelcomeEndpoints();

// 在背景執行 Web API
var webTask = webApp.RunAsync();

// 建立 Console 應用程式
var builder = Host.CreateApplicationBuilder(args);
builder.Services.AddSingleton<Main>();
var host = builder.Build();

// 執行 Console 應用程式
var main = host.Services.GetRequiredService<Main>();
main.Run();

// 等待 Web API 完成
await webTask;

