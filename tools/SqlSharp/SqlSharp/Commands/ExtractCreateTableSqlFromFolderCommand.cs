using Microsoft.Extensions.Logging;
using SqlSharp.CommandPattern;
using SqlSharpLit.Common.ParserLit;

namespace SqlSharp.Commands;

public class ExtractCreateTableSqlFromFolderCommand : ISpecificationAsync<SqlSharpOptions, Task>
{
    private ILogger<ExtractCreateTableSqlFromFolderCommand> _logger;
    public ExtractCreateTableSqlFromFolderCommand(ILogger<ExtractCreateTableSqlFromFolderCommand> logger)
    {
        _logger = logger;
    }
    
    public bool IsMatch(SqlSharpOptions args)
    {
        return args.IsActionName(SqlSharpOptions.ExtractCreateTableSql);
    }

    public Task<Task> ExecuteAsync(SqlSharpOptions args)
    {
        _logger.LogInformation("ExtractCreateTableSqlFromFolderCommand.ExecuteAsync");
        var sourceFolder = args.Input;
        var outputFolder = args.Output;
        var extractSqlHelper = new ExtractSqlHelper(new CustomDatabaseNameProvider());

        extractSqlHelper.SetDatabaseNameDeep(args.DatabaseNamePathDeep);
        extractSqlHelper.GenerateDatabasesDescriptionJsonFileFromFolder(sourceFolder, outputFolder);
        extractSqlHelper.UpdateByUserDatabasesDescription(outputFolder);
        return Task.FromResult(Task.CompletedTask);
    }
}