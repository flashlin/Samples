using System.Globalization;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;
using SqlSharp.CommandPattern;
using SqlSharpLit;

namespace SqlSharp;

public class ExtractTableDataCommand : ICommand<SqlSharpOptions>
{
    private DynamicDbContext _db;

    public ExtractTableDataCommand(DynamicDbContext db)
    {
        _db = db;
    }

    public ICommand<SqlSharpOptions>? Next { get; set; }

    public async Task ExecuteAsync(SqlSharpOptions options)
    {
        if (!options.IsActionName(SqlSharpOptions.ExportTableData))
        {
            await Next.SafeExecuteAsync(options);
            return;
        }

        var sourceTable = options.Input;
        var targetCsvFile = options.Output;
        var records = _db.ExportTableData(sourceTable);
        var csvWriter = new CCsvWriter();
        await csvWriter.WriteRecordsAsync(records, targetCsvFile);
    }
}

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