using Microsoft.Extensions.Logging;
using SqlSharp.CommandPattern;
using SqlSharpLit.Common.ParserLit;

namespace SqlSharp.Commands;

public class ExtractSelectSqlFromFolderCommand : ISpecificationAsync<SqlSharpOptions, Task>
{
    private ILogger<ExtractSelectSqlFromFolderCommand> _logger;
    public ExtractSelectSqlFromFolderCommand(ILogger<ExtractSelectSqlFromFolderCommand> logger)
    {
        _logger = logger;
    }
    
    public bool IsMatch(SqlSharpOptions args)
    {
        return args.IsActionName(SqlSharpOptions.ExtractSelectSql);
    }

    public async Task<Task> ExecuteAsync(SqlSharpOptions args)
    {
        var sourceFolder = args.Input;
        var outputFolder = args.Output;
        var extractSqlHelper = new ExtractSqlHelper(new CustomDatabaseNameProvider());
        extractSqlHelper.SetDatabaseNameDeep(args.DatabaseNamePathDeep);
        
        //extractSqlHelper.ExtractSelectSqlFromFolder(sourceFolder, outputFolder);
        
        await extractSqlHelper.GenerateSelectStatementQaMdFileAsync(sourceFolder, outputFolder);
        
        return Task.FromResult(Task.CompletedTask);
    }
}