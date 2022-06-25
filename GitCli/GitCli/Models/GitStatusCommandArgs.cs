namespace GitCli.Models;

public class GitStatusCommandArgs
{
    [Value(index: 0, HelpText = "action name")]
    public string ActionName { get; set; } = "";
}