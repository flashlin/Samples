using SqlSharp.CommandPattern;
using SqlSharpLit.Common.ParserLit;

namespace SqlSharp;

public class ExtractCreateTableSqlFromFolderCommand : ISpecificationAsync<SqlSharpOptions, Task>
{
    public bool IsMatch(SqlSharpOptions args)
    {
        return args.IsActionName(SqlSharpOptions.ExtractCreateTableSql);
    }

    public Task<Task> ExecuteAsync(SqlSharpOptions args)
    {
        var sourceFolder = args.Input;
        var outputFolder = args.Output;
        var extractSqlHelper = new ExtractSqlHelper(new CustomDatabaseNameProvider());
        extractSqlHelper.WriteCreateTablesFromFolder(sourceFolder, outputFolder);
        return Task.FromResult(Task.CompletedTask);
    }
}