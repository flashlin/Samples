using CommandLine;
using Microsoft.Extensions.Logging;

namespace SqlSharp;

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