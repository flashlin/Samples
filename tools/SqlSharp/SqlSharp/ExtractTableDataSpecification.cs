using SqlSharp.CommandPattern;
using SqlSharpLit;

namespace SqlSharp;

public class ExtractTableDataSpecification : ISpecification<SqlSharpOptions, Task>
{
    private DynamicDbContext _db;

    public ExtractTableDataSpecification(DynamicDbContext db)
    {
        _db = db;
    }

    public bool IsMatch(SqlSharpOptions options)
    {
        return options.IsActionName(SqlSharpOptions.ExportTableData);
    }

    public async Task<Task> ExecuteAsync(SqlSharpOptions options)
    {
        var sourceTable = options.Input;
        var targetCsvFile = options.Output;
        var records = _db.ExportTableData(sourceTable);
        var csvWriter = new CCsvWriter();
        await csvWriter.WriteRecordsAsync(records, targetCsvFile);
        return Task.CompletedTask;
    }
}