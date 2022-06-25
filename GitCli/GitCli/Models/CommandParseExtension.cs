using CommandLine;

namespace GitCli.Models;

public static class CommandParseExtension
{
    public static T? ParseArgs<T>(this string[] args)
    {
        var opts = Parser.Default.ParseArguments<T>(args)
            .MapResult((opts) => opts,
                errs => default(T));
        return opts;
    }
}