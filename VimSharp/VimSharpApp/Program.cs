// See https://aka.ms/new-console-template for more information
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using VimSharpApp;
using VimSharpApp.ApiHandlers;

namespace VimSharpApp
{
    // Program類別
    public class Program
    {
        // 應用程序入口點
        public static async Task Main(string[] args)
        {
            // 建立及啟動 Web 應用程式
            var webAppInstance = new WebApp();
            await webAppInstance.StartAsync(args);

            if(Console.IsInputRedirected)
            {
                // 等待 Web 應用程式結束
                await webAppInstance.WaitForShutdownAsync();
            }

            // 如果沒有輸入重定向，則建立及執行 Console 應用程式
            if(!Console.IsInputRedirected)
            {
                // 建立及執行 Console 應用程式
                var consoleAppInstance = new ConsoleApp();
                consoleAppInstance.Start(args);

                // 關閉 Web 應用程式
                await webAppInstance.Shutdown();
            }
        }
    }
}

