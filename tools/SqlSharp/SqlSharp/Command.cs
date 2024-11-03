using CommandLine;
using Microsoft.Extensions.Logging;

namespace SqlSharp;

public class SqlSharpOptions
{
    [Option('v', "Verb", Required = true, HelpText = "The action to perform. (export)")]
    public string ActionName { get; set; } = string.Empty;
    
    [Option('i', "Input", Required = false, HelpText = "Input file or folder")]
    public string Input { get; set; } = "";
    
    [Option('o', "Output", Required = false, HelpText = "Output file or folder")]
    public string Output { get; set; } = "";
    
    public bool IsActionName(string expectedActionName)
    {
        return ActionName.Equals(expectedActionName, StringComparison.InvariantCultureIgnoreCase);
    }
}

public static class LineCommandParseHelper
{
    public static Task<T> ParseAsync<T>(string[] args)
    {
        var promise = new TaskCompletionSource<T>();
        Parser.Default.ParseArguments<T>(args)
            .WithParsed(options =>
            {
                LoggerFactory.Create(builder =>
                {
                    builder
                        .ClearProviders()
                        .AddConsole();
                });
                promise.TrySetResult(options);
            })
            .WithNotParsed(errors =>
            {
                foreach (var error in errors)
                {
                    Console.WriteLine($"Error: {error}");
                }
                promise.TrySetException(new Exception("Invalid command line arguments"));
            });
        return promise.Task;
    }
}

public interface IArgumentCommand
{
    IArgumentCommand? Next { get; set; } 
    Task ExecuteAsync(SqlSharpOptions options);
}