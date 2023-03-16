using System.Text.Json;

namespace QueryKits.ExcelUtils;

public static class JsonHelper
{
    public static string ToCsvString(this string json)
    {
        var dataList = JsonSerializer.Deserialize<List<Dictionary<string, object>>>(json);
        if (dataList == null || dataList.Count == 0)
        {
            return string.Empty;
        }
        var csvWriter = new CsvMemoryWriter();
        var headers = dataList[0].Keys.ToList();
        csvWriter.WriteHeaders(headers);
        foreach (var row in dataList)
        {
            var newRow = row.ToDictionary(x => x.Key, x => $"{x.Value}");
            csvWriter.WriteRow(newRow);
        }
        return csvWriter.ToCsvString();
    }
}