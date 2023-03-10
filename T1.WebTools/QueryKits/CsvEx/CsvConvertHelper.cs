using System.Globalization;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;

namespace QueryKits.CsvEx;

public static class CsvConvertHelper
{
    public static void WriteToFile(this Stream stream, string outputFile)
    {
        using var outputStream = new FileStream(outputFile, FileMode.Create, FileAccess.Write);
        using var writer = new StreamWriter(outputStream, Encoding.UTF8);
        using var reader = new StreamReader(stream, Encoding.UTF8);
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line == null)
            {
                break;
            }
            writer.WriteLine(line);
        }
        writer.Flush();
        outputStream.Close();
    }
    
    
    public static void ToCsvFile(this List<Dictionary<string, object>> dataList, string outputFile)
    {
        if (dataList.Count == 0)
        {
            return;
        }
        var headers = dataList[0].Keys.ToList();
        var csvConfig = new CsvConfiguration(CultureInfo.CurrentCulture)
        {
            HasHeaderRecord = true,
            Delimiter = ",",
            Encoding = Encoding.UTF8
        };
        using var stream = new FileStream(outputFile, FileMode.Create);
        using var writer = new StreamWriter(stream);
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
    }
    
    
    public static MemoryStream ToCsvStream(this List<Dictionary<string, object>> dataList)
    {
        var memoryStream = new MemoryStream();
        if (dataList.Count == 0)
        {
            return memoryStream;
        }
        var headers = dataList[0].Keys.ToList();
        var csvConfig = new CsvConfiguration(CultureInfo.CurrentCulture)
        {
            HasHeaderRecord = true,
            Delimiter = ",",
            Encoding = Encoding.UTF8,
        };
        var writer = new StreamWriter(memoryStream);
        using var csvWriter = new CsvWriter(writer, csvConfig, true);

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
        csvWriter.Flush();
        memoryStream.Position = 0;
        return memoryStream;
    }

    public static CsvSheet ToCsvSheet(this Stream stream, string delimiter=",")
    {
        return CsvSheet.ReadFromStream(stream, delimiter);
    }

    public static List<string> ToCsvMultiLines(this Stream csvStream)
    {
        using var reader = new StreamReader(csvStream, Encoding.UTF8);
        var result = new List<string>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line == null)
            {
                break;
            }
            result.Add(line);
        }

        return result;
    }
}