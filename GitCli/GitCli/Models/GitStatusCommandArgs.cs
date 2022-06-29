using CommandLine;

namespace GitCli.Models;

[Verb("s", HelpText = "status")]
public class GitStatusCommandArgs
{
    [Value(index: 0, HelpText = "action name")]
    public string ActionName { get; set; } = "";
}