namespace GitCli.Models;

public class GitRepoInfo
{
    public List<FileStatusInfo> Status { get; set; } = new List<FileStatusInfo>();
    public string FolderPath { get; init; }
}