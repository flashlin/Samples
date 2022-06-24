namespace GitCli.Models;

public class FileStatusInfo
{
    public string FilePath { get; init; }
    public GitFileStatus Status { get; set; }
}