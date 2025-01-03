using System.Globalization;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;

namespace SqlSharpLit.Common;

public class CsvSharpReader : IDisposable 
{
    private readonly CsvConfiguration _csvConfig = new CsvConfiguration(CultureInfo.InvariantCulture)
    {
        HasHeaderRecord = true
    };
    private StreamReader? _reader;
    private CsvReader? _csv;

    public Task OpenFileAsync<T>(string csvFile)
    {
        if (!File.Exists(csvFile))
        {
            throw new FileNotFoundException($"{csvFile}");
        }
        _reader = new StreamReader(csvFile, Encoding.UTF8);
        _csv = new CsvReader(_reader, _csvConfig);
        return Task.CompletedTask;
    }

    public async IAsyncEnumerable<T> ReadRecordsAsync<T>(Func<T, T> createRecordFn)
        where T: class, new()
    {
        if (_csv == null)
        {
            throw new InvalidOperationException("CsvReader is not opened");
        }
        var item = new T();
        await foreach (var data in _csv.EnumerateRecordsAsync<T>(item))
        {
            yield return createRecordFn(data);
        }
    }

    public void Close()
    {
        _csv?.Dispose();
        _reader?.Dispose();
        _csv = null;
        _reader = null;
    }

    public void Dispose()
    {
        Close();
    }
}

public class CsvSharpWriter : IDisposable
{
    private readonly CsvConfiguration _csvConfig = new CsvConfiguration(CultureInfo.InvariantCulture)
    {
        HasHeaderRecord = true
    };
    private bool _isExisted;
    private CsvWriter? _csv;
    private StreamWriter? _writer;

    public async Task CreateFileAsync<T>(string csvFile)
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

    public void Close()
    {
        _csv?.Dispose();
        _csv = null;
    }

    public void Dispose()
    {
        Close();
    }
}