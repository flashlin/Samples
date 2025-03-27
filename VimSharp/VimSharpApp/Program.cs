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
            // 主程式
            var webApp = WebApp.CreateWebApplication(args);
            var webTask = webApp.RunAsync();

            var main = ConsoleApp.CreateConsoleApplication(args);
            main.Run();

            // 關閉 Web 應用程式
            await WebApp.Shutdown(webApp, webTask);
        }
    }
}

