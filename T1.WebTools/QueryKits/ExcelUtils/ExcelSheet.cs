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

    public string ToCsv()
    {
        var csvConfig = new CsvConfiguration(CultureInfo.CurrentCulture)
        {
            HasHeaderRecord = true,
            Delimiter = ",",
            Encoding = Encoding.UTF8
        };
        using var mem = new MemoryStream();
        using var writer = new StreamWriter(mem);
        using var csvWriter = new CsvWriter(writer, csvConfig);
        foreach (var header in Headers)
        {
            csvWriter.WriteField(header);
        }

        csvWriter.NextRecord();
        foreach (var row in Rows)
        {
            foreach (var header in Headers)
            {
                csvWriter.WriteField(row[header.Name]);
            }

            csvWriter.NextRecord();
        }

        writer.Flush();
        mem.Position = 0;
        using var sr = new StreamReader(mem);
        return sr.ReadToEnd();
    }
}