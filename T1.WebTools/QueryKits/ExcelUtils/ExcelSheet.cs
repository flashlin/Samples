using System.Globalization;
using System.Text;
using System.Text.Json;
using CsvHelper;
using CsvHelper.Configuration;

namespace QueryKits.ExcelUtils;

public class ExcelSheet
{
    public string Name { get; set; } = string.Empty;
    public List<ExcelColumn> Headers { get; set; } = new();
    public List<Dictionary<string, string>> Rows { get; set; } = new();

    public string ToCsvString()
    {
        using var csvWriter = new CsvMemoryWriter();
        csvWriter.WriteHeaders(Headers.Select(x => x.Name));
        foreach (var row in Rows)
        {
            csvWriter.WriteRow(row);
        }
        return csvWriter.ToCsvString();
    }
}