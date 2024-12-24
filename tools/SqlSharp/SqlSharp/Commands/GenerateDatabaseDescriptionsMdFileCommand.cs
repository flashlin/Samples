using Microsoft.Extensions.Logging;
using SqlSharp.CommandPattern;
using SqlSharpLit.Common.ParserLit;

namespace SqlSharp.Commands;

public class GenerateDatabaseDescriptionsMdFileCommand : ISpecificationAsync<SqlSharpOptions, Task>
{
    private readonly ILogger<GenerateDatabaseDescriptionsMdFileCommand> _logger;
    
    public GenerateDatabaseDescriptionsMdFileCommand(ILogger<GenerateDatabaseDescriptionsMdFileCommand> logger)
    {
        _logger = logger;
    }
    
    public bool IsMatch(SqlSharpOptions args)
    {
        return args.IsActionName(SqlSharpOptions.GenerateDatabaseDescriptionsMdFile);
    }

    public Task<Task> ExecuteAsync(SqlSharpOptions args)
    {
        var databaseDescriptionJsonFile = args.Input;
        var extractSqlHelper = new ExtractSqlHelper(new CustomDatabaseNameProvider());
        extractSqlHelper.GenerateDatabaseDescriptionsQaMdFile(databaseDescriptionJsonFile);
        return Task.FromResult(Task.CompletedTask);
    }
}