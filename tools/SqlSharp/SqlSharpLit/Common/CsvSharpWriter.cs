using System.Globalization;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;

namespace SqlSharpLit.Common;

public class CsvSharpWriter : IDisposable
{
    private readonly CsvConfiguration _csvConfig = new CsvConfiguration(CultureInfo.InvariantCulture)
    {
        HasHeaderRecord = true
    };
    private bool _isExisted;
    private CsvWriter? _csv;
    private StreamWriter? _writer;

    public async Task CreateAsync<T>(string csvFile)
    {
        _isExisted = File.Exists(csvFile);
        _writer = new StreamWriter(csvFile, Encoding.UTF8, new FileStreamOptions
        {
            Access = FileAccess.Write,
            Mode = _isExisted ? FileMode.Append : FileMode.Create, 
        });
        _csv = new CsvWriter(_writer, _csvConfig);
        if (!_isExisted)
        {
            _csv.WriteHeader<T>();
            await _csv.NextRecordAsync();
        }
    }

    public async Task WriteRecordAsync<T>(T record)
    {
        if (_csv == null)
        {
            throw new InvalidOperationException("CsvWriter is not created");
        }
        _csv.WriteRecord(record);
        await _csv.NextRecordAsync();
    }

    public async Task FlushAsync()
    {
        if (_csv != null)
        {
            await _csv.FlushAsync();
        }
    }

    public void Dispose()
    {
        _csv?.Dispose();
    }
}