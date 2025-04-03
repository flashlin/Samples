using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using VimSharpApp;
using VimSharpLib;

namespace VimSharpApp
{
    // ConsoleApp類別
    public class ConsoleApp
    {
        private Main _main;

        // 建立並執行 Console 應用程式的方法
        public void Start(string[] args)
        {
            var builder = Host.CreateApplicationBuilder(args);

            var services = builder.Services;
            services.AddVimSharpServices();
            services.AddSingleton<Main>();

            var host = builder.Build();

            // 取得 Console 應用程式的實例
            _main = host.Services.GetRequiredService<Main>();

            // 執行 Console 應用程式
            _main.Run();
        }
    }
}    
     