using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using VimSharpApp;

namespace VimSharpApp
{
    // ConsoleApp類別
    public static class ConsoleApp
    {
        // 建立 Console 應用程式的方法
        public static Main CreateConsoleApplication(string[] args)
        {
            var builder = Host.CreateApplicationBuilder(args);
            builder.Services.AddSingleton<Main>();
            var host = builder.Build();

            // 取得 Console 應用程式的實例
            return host.Services.GetRequiredService<Main>();
        }
    }
} 