// See https://aka.ms/new-console-template for more information
// 引入 Everything SDK 函數

using System.Runtime.InteropServices;
using System.Text;
using EverythingDemo;

var regexPattern = ".*\\.txt";  // 例：搜尋所有 .txt 檔案
var everything = new Everything();
foreach (var file in everything.SearchByRegex(regexPattern))
{
    Console.WriteLine(file);  // 輸出每個檔案的完整路徑
}
Console.WriteLine("END!");