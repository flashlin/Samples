using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System;
using VimSharpLib;

namespace VimSharp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = Host.CreateApplicationBuilder(args);
            
            // 註冊服務
            builder.Services.AddSingleton<IConsoleDevice, ConsoleDeviceAdapter>();
            builder.Services.AddSingleton<VimEditor>();
            
            var host = builder.Build();
            
            // 獲取 VimEditor 實例
            var editor = host.Services.GetRequiredService<VimEditor>();
            
            // 讀取命令行參數中的文件名
            string filename = args.Length > 0 ? args[0] : "";
            if (!string.IsNullOrEmpty(filename))
            {
                editor.OpenFile(filename);
            }
            
            // 初始化編輯器的模式
            editor.Mode = new VimNormalMode { Instance = editor };
            
            Console.Clear();
            Console.CursorVisible = true;
            
            // 主循環
            while (true)
            {
                try
                {
                    editor.Render();
                    editor.Mode.WaitForInput();
                }
                catch (Exception ex)
                {
                    editor.StatusBar = new ConsoleText($"Error: {ex.Message}");
                    editor.Render();
                    Console.ReadKey(true);
                }
            }
        }
    }
}
