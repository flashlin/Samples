using System;
using System.Threading.Tasks;

namespace ConsoleTestApp
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("測試 gRPC 服務生成器...");
            
            // 創建 MyGreeter 的實例
            var greeter = new MyGreeter();
            
            // 測試 SayHelloAsync 方法
            string helloResult = await greeter.SayHelloAsync("張三");
            Console.WriteLine($"SayHelloAsync 結果: {helloResult}");
            
            // 測試 GetUserInfoAsync 方法
            UserInfo userInfo = await greeter.GetUserInfoAsync(42, true);
            Console.WriteLine($"GetUserInfoAsync 結果:");
            Console.WriteLine($"  ID: {userInfo.Id}");
            Console.WriteLine($"  姓名: {userInfo.Name}");
            Console.WriteLine($"  年齡: {userInfo.Age}");
            Console.WriteLine($"  詳細信息: {userInfo.Details}");
            
            // 測試 Add 方法
            int sum = greeter.Add(10, 20);
            Console.WriteLine($"Add 結果: {sum}");
            
            Console.WriteLine("按任意鍵退出...");
            Console.ReadKey();
        }
    }
}
