using SqlSharp.CommandPattern;

namespace SqlSharp;

public class ExtractTableDataCommand : ICommand<SqlSharpOptions>
{

    public ICommand<SqlSharpOptions>? Next { get; set; }

    public async Task ExecuteAsync(SqlSharpOptions options)
    {
        if (!options.IsActionName(SqlSharpOptions.ExportTableData))
        {
            await Next.SafeExecuteAsync(options);
            return;
        }       
    }
}