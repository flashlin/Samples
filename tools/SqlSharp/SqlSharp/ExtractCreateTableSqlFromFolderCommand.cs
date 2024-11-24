using SqlSharp.CommandPattern;
using SqlSharpLit.Common.ParserLit;

namespace SqlSharp;

public class ExtractCreateTableSqlFromFolderCommand : ICommand<SqlSharpOptions>
{
    public ICommand<SqlSharpOptions>? Next { get; set; }
    public async Task ExecuteAsync(SqlSharpOptions args)
    {
        if(!args.IsActionName(SqlSharpOptions.ExtractCreateTableSql))
        {
            await Next.SafeExecuteAsync(args);
            return;
        }

        var sourceFolder = args.Input;
        var outputFolder = args.Output;
        var extractSqlHelper = new ExtractSqlHelper(new CustomDatabaseNameProvider());
        extractSqlHelper.WriteCreateTablesFromFolder(sourceFolder, outputFolder);
        extractSqlHelper.GenerateRagFiles(sourceFolder);
    }
}