using System.Globalization;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;

namespace SqlSharp;

public class CCsvWriter
{
    public async Task WriteRecordsAsync<T>(IEnumerable<T> records, string outputCsvFile)
    {
        var csvConfig = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true
        };
        await using var writer = new StreamWriter(outputCsvFile, Encoding.UTF8, new FileStreamOptions());
        await using var csv = new CsvWriter(writer, csvConfig);
        await csv.WriteRecordsAsync(records);
    }
}