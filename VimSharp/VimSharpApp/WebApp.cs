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
        private readonly string _staticFolder = "wwwroot"; // 靜態檔案目錄

        // 建立並啟動 Web 應用程式的方法
        public async Task StartAsync(string[] args)
        {
            var webBuilder = WebApplication.CreateBuilder(args);

            // 設定日誌，移除 Console 輸出
            webBuilder.Logging.ClearProviders();
            webBuilder.Logging.AddDebug();
            webBuilder.Logging.AddConsole();

            // 載入 appSetting.json
            webBuilder.Configuration.AddJsonFile("appSetting.json", optional: false, reloadOnChange: true);
            webBuilder.Services.Configure<AppSettingConfig>(webBuilder.Configuration);

            // 加入 Swagger/OpenAPI 服務
            webBuilder.Services.AddEndpointsApiExplorer();
            webBuilder.Services.AddSwaggerGen();
            webBuilder.Services.AddHttpClient();
            var services = webBuilder.Services;
            services.AddSingleton<ITestApiHandler, TestApiHandler>();

            webBuilder.WebHost.UseUrls("http://*:8080");
            _webApp = webBuilder.Build();

            // 啟用 Swagger UI（僅限開發環境）
            //if (_webApp.Environment.IsDevelopment())
            {
                _webApp.UseSwagger();
                _webApp.UseSwaggerUI();
            }

            // 註冊 API 端點
            JobApiHandler.MapEndpoints(_webApp);
            TestApiHandler.MapEndpoints(_webApp);

            // 提供 wwwroot 資料夾的靜態檔案
            _webApp.UseDefaultFiles(new DefaultFilesOptions
            {
                FileProvider = new Microsoft.Extensions.FileProviders.PhysicalFileProvider(System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), _staticFolder)),
                RequestPath = ""
            });
            _webApp.UseStaticFiles(new StaticFileOptions
            {
                FileProvider = new Microsoft.Extensions.FileProviders.PhysicalFileProvider(System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), _staticFolder)),
                RequestPath = ""
            });
            
            // 啟動 Web 應用程式
            _webTask = _webApp.RunAsync();
        }
        
        // 關閉 Web 應用程式的方法
        public async Task Shutdown()
        {
            // 在主程式結束時觸發 Web 應用程式的關閉
            _webApp.Lifetime.StopApplication();

            // 等待 Web API 完成
            await _webTask;
        }

        // 等待 Web 應用程式結束的方法
        public async Task WaitForShutdownAsync()
        {
            if (_webTask != null)
            {
                await _webTask;
            }
        }
    }
} 