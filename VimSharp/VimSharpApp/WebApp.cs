using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Logging;
using VimSharpApp.ApiHandlers;
using System.Threading.Tasks;

namespace VimSharpApp
{
    // WebApp類別
    public class WebApp
    {
        private WebApplication _webApp;
        private Task _webTask;

        // 建立 Web 應用程式的方法
        public WebApplication CreateWebApplication(string[] args)
        {
            var webBuilder = WebApplication.CreateBuilder(args);

            // 設定日誌，移除 Console 輸出
            webBuilder.Logging.ClearProviders();
            webBuilder.Logging.AddDebug();

            webBuilder.WebHost.UseUrls("http://localhost:8080");
            _webApp = webBuilder.Build();

            // 註冊 API 端點
            JobApiHandler.MapEndpoints(_webApp);
            
            return _webApp;
        }
        
        // 啟動 Web 應用程式
        public Task StartAsync()
        {
            _webTask = _webApp.RunAsync();
            return _webTask;
        }
        
        // 關閉 Web 應用程式的方法
        public async Task Shutdown()
        {
            // 在主程式結束時觸發 Web 應用程式的關閉
            _webApp.Lifetime.StopApplication();

            // 等待 Web API 完成
            await _webTask;
        }
    }
} 