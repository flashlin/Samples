using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Logging;
using VimSharpApp.ApiHandlers;
using System.Threading.Tasks;

namespace VimSharpApp
{
    // WebApp類別
    public static class WebApp
    {
        // 建立 Web 應用程式的方法
        public static WebApplication CreateWebApplication(string[] args)
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
        
        // 關閉 Web 應用程式的方法
        public static async Task Shutdown(WebApplication webApp, Task webTask)
        {
            // 在主程式結束時觸發 Web 應用程式的關閉
            webApp.Lifetime.StopApplication();

            // 等待 Web API 完成
            await webTask;
        }
    }
} 