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


public static class DictListExtension
{
    public static string ToCsv(IEnumerable<Dictionary<string, string>> dictList)
    {
        var index = 0;
        var headers = new List<string>();
        using var csvWriter = CreateCsvWriter();
        foreach (var item in dictList)
        {
            if (index == 0)
            {
                headers = item.Keys.ToList();
                csvWriter.WriteHeaders(headers);
            }
            csvWriter.WriteRow(headers, item);
        }
        
    }

    public static CsvWriter CreateCsvWriter()
    {
        var csvConfig = new CsvConfiguration(CultureInfo.CurrentCulture)
        {
            HasHeaderRecord = true,
            Delimiter = ",",
            Encoding = Encoding.UTF8
        };
        var mem = new MemoryStream();
        var writer = new StreamWriter(mem);
        var csvWriter = new CsvWriter(writer, csvConfig);
        return csvWriter;
    }

    public static void WriteHeaders(this CsvWriter csvWriter, IEnumerable<string> headers)
    {
        foreach (var header in headers)
        {
            csvWriter.WriteField(header);
        }
        csvWriter.NextRecord();
    }

    public static void WriteRow(this CsvWriter csvWriter, IEnumerable<string> headers, Dictionary<string, string> row)
    {
        foreach (var header in headers)
        {
            csvWriter.WriteField(row[header]);
        }
        csvWriter.NextRecord();
    }

    public static void ToString(this CsvWriter csvWriter)
    {
        csvWriter.Flush();
        mem.Position = 0;
        using var sr = new StreamReader(mem);
        return sr.ReadToEnd();
    }
    
    public static string ToCsv(this string json)
    {
        var dataList = JsonSerializer.Deserialize<List<Dictionary<string, dynamic>>>(json);
        if (dataList == null || dataList.Count == 0)
        {
            return string.Empty;
        }

        var headers = dataList[0].Keys.ToList();

        var csvConfig = new CsvConfiguration(CultureInfo.CurrentCulture)
        {
            HasHeaderRecord = true,
            Delimiter = ",",
            Encoding = Encoding.UTF8
        };
        using var mem = new MemoryStream();
        using var writer = new StreamWriter(mem);
        using var csvWriter = new CsvWriter(writer, csvConfig);

        foreach (var header in headers)
        {
            csvWriter.WriteField(header);
        }

        csvWriter.NextRecord();

        foreach (var row in dataList)
        {
            foreach (var header in headers)
            {
                csvWriter.WriteField(row[header]);
            }

            csvWriter.NextRecord();
        }

        writer.Flush();
        mem.Position = 0;
        using var sr = new StreamReader(mem);
        return sr.ReadToEnd();

    }
}