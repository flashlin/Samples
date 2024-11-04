using System.Globalization;
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
        CsvHelper.Configuration.CsvConfiguration csvConfig = new(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true
        };
    }
}