using System;
using System.IO;

namespace GenGrpcService
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // 顯示應用程式標題
            Console.WriteLine("====================================");
            Console.WriteLine("   gRPC 服務代碼生成工具   ");
            Console.WriteLine("====================================");
            Console.WriteLine();

            // 檢查命令行參數
            if (args.Length < 1)
            {
                Console.WriteLine("用法：GenGrpcService <專案文件路徑>");
                Console.WriteLine("範例：GenGrpcService D:\\Projects\\MyApi\\MyApi.csproj");
                Console.WriteLine();
                
                // 互動式模式：如果沒有提供命令行參數，則提示用戶輸入
                Console.Write("請輸入專案文件路徑: ");
                string projectFilePath = Console.ReadLine().Trim();
                
                if (string.IsNullOrEmpty(projectFilePath))
                {
                    Console.WriteLine("未提供專案文件路徑，程式退出。");
                    return;
                }
                
                if (!File.Exists(projectFilePath))
                {
                    Console.WriteLine($"錯誤：找不到專案文件 {projectFilePath}");
                    return;
                }
                
                ExecuteGenerator(projectFilePath);
            }
            else
            {
                string projectFilePath = args[0];
                
                if (!File.Exists(projectFilePath))
                {
                    Console.WriteLine($"錯誤：找不到專案文件 {projectFilePath}");
                    return;
                }
                
                ExecuteGenerator(projectFilePath);
            }
        }
        
        private static void ExecuteGenerator(string projectFilePath)
        {
            try
            {
                Console.WriteLine($"開始處理專案: {projectFilePath}");
                Console.WriteLine();
                
                // 執行代碼生成
                var generator = new GrpcServiceSourceGenerator();
                generator.Execute(projectFilePath);
                
                Console.WriteLine();
                Console.WriteLine("代碼生成完成！");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"發生錯誤: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
            
            Console.WriteLine();
            Console.WriteLine("按任意鍵退出...");
            Console.ReadKey();
        }
    }
} 