using System.Text;
using CsvHelper;
using System.Globalization;
using System.IO;

namespace VimSharpLib;

public class TextToCsvConverter
{
    private readonly char[] _delimiters = [',', ';', '\t'];
    public string Convert(string text)
    {
        var lines = text.Split('\n')
            .Select(l => l.TrimEnd('\r'))
            .Take(3)
            .ToList();

        if (lines.Count == 0) return string.Empty;
        
        var selectedDelimiter = GetSelectedDelimiter(lines);
        // 分割第一行作為欄位名稱
        var headers = GetHeaders(lines, selectedDelimiter);
        // 準備資料列
        var dataRows = GetDataRows(lines, selectedDelimiter);

        // 使用 CsvHelper 建立 CSV 字串
        using var memoryStream = new MemoryStream();
        using var writer = new StreamWriter(memoryStream);
        using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
        // 寫入標題
        WriteHeaders(headers, csv);
        // 寫入資料列
        WriteDataRows(dataRows, csv);
        writer.Flush();
        return Encoding.UTF8.GetString(memoryStream.ToArray());
    }

    private static void WriteDataRows(List<List<string>> dataRows, CsvWriter csv)
    {
        foreach (var row in dataRows)
        {
            foreach (var field in row)
            {
                csv.WriteField(field);
            }
            csv.NextRecord();
        }
    }

    private static void WriteHeaders(List<string> headers, CsvWriter csv)
    {
        foreach (var header in headers)
        {
            csv.WriteField(header);
        }

        csv.NextRecord();
    }

    private static List<string> GetHeaders(List<string> lines, char selectedDelimiter)
    {
        var headers = lines[0].Split(selectedDelimiter)
            .Select(h => h.Trim())
            .ToList();
        return headers;
    }

    private static List<List<string>> GetDataRows(List<string> lines, char selectedDelimiter)
    {
        var dataRows = lines.Skip(1)
            .Select(line => line.Split(selectedDelimiter)
                .Select(field => field.Trim())
                .ToList())
            .ToList();
        return dataRows;
    }

    private char GetSelectedDelimiter(List<string> lines)
    {
        // 計算前三行中每個分隔符的數量
        var delimiterCounts = ComputeDelimiterCounts(lines);
        // 找出在所有行中數量相同的分隔符
        var selectedDelimiter = ComputeSelectedDelimiter(delimiterCounts);
        return selectedDelimiter;
    }

    private char ComputeSelectedDelimiter(Dictionary<char, List<int>> delimiterCounts)
    {
        char selectedDelimiter = _delimiters[0]; // 預設使用第一個分隔符
        foreach (var kvp in delimiterCounts)
        {
            var counts = kvp.Value;
            if (counts.Count > 1 && counts.All(c => c == counts[0]) && counts[0] > 0)
            {
                selectedDelimiter = kvp.Key;
                break;
            }
        }

        return selectedDelimiter;
    }

    private Dictionary<char, List<int>> ComputeDelimiterCounts(List<string> lines)
    {
        var delimiterCounts = new Dictionary<char, List<int>>();
        foreach (var delimiter in _delimiters)
        {
            delimiterCounts[delimiter] = new List<int>();
        }

        foreach (var line in lines)
        {
            foreach (var delimiter in _delimiters)
            {
                delimiterCounts[delimiter].Add(line.Count(c => c == delimiter));
            }
        }

        return delimiterCounts;
    }
}