namespace GitCli.Models;

public class FileStatusInfo
{
    public string FilePath { get; init; } = string.Empty;
    public GitFileStatus Status { get; set; }
}