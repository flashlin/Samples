using System.Globalization;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;

namespace QueryKits.ExcelUtils;

public class CsvMemoryWriter : IDisposable
{
    private readonly CsvWriter _csvWriter;
    private readonly MemoryStream _memoryStream;
    private List<string> _headers = new();

    public CsvMemoryWriter()
    {
        _memoryStream = new MemoryStream();
        _csvWriter = new CsvWriter(new StreamWriter(_memoryStream), new CsvConfiguration(CultureInfo.CurrentCulture)
        {
            HasHeaderRecord = true,
            Delimiter = ",",
            Encoding = Encoding.UTF8
        });
    }

    public void Dispose()
    {
        _memoryStream.Dispose();
        _csvWriter.Dispose();
    }

    public void WriteHeaders(IEnumerable<string> headers)
    {
        _headers = headers.ToList();
        foreach (var header in _headers)
        {
            _csvWriter.WriteField(header);
        }
        _csvWriter.NextRecord();
    }

    public void WriteRow(Dictionary<string, string> row)
    {
        var hasHeaders = false;
        foreach (var header in _headers)
        {
            _csvWriter.WriteField(row[header]);
            hasHeaders = true;
        }

        if (!hasHeaders)
        {
            foreach (var header in row.Keys)
            {
                _csvWriter.WriteField(row[header]);
            }
        }
        
        _csvWriter.NextRecord();
    }

    public string ToCsvString()
    {
        _csvWriter.Flush();
        _memoryStream.Position = 0;
        using var sr = new StreamReader(_memoryStream);
        return sr.ReadToEnd();
    }
}