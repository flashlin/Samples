using System.Runtime.InteropServices;
using System.Text;

namespace EverythingDemo;

public class Everything
{
    [DllImport("Everything64.dll", CharSet = CharSet.Unicode)]
    private static extern int Everything_SetSearchW(string search);
    
    [DllImport("Everything64.dll")]
    private static extern void Everything_SetRegex(bool isRegex);
    
    [DllImport("Everything64.dll")]
    private static extern bool Everything_QueryW(bool wait);
    
    [DllImport("Everything64.dll")]
    private static extern int Everything_GetNumResults();
    
    [DllImport("Everything64.dll", CharSet = CharSet.Unicode)]
    private static extern int Everything_GetResultFullPathName(int index, StringBuilder buf, int bufSize);

    public List<string> SearchByRegex(string regexPattern)
    {
        Everything_SetSearchW(regexPattern); // 設定搜尋字串
        Everything_SetRegex(true);           // 啟用正則表達式搜尋
        Everything_QueryW(true);         // 開始查詢
        var numResults = Everything_GetNumResults();
        var fileList = new List<string>();
        for (int i = 0; i < numResults; i++)
        {
            var resultPath = new StringBuilder(260);  // 檔案路徑 buffer
            Everything_GetResultFullPathName(i, resultPath, resultPath.Capacity);
            fileList.Add(resultPath.ToString());  // 輸出每個檔案的完整路徑
        }
        return fileList;
    }
}