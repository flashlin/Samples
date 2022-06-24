using LibGit2Sharp;

namespace GitCli.Models;

public class GitRepoInfo
{
    public string FolderPath { get; init; }
    public IEnumerable<FileStatusInfo> QueryStatus()
    {
        using (var repo = new Repository(FolderPath))
        {
            //var master = repo.Branches["master"];
            var status = repo.RetrieveStatus();
            //status.IsDirty;
            //status.Modified
            foreach (var modified in status.Modified)
            {
                yield return new FileStatusInfo
                {
                    FilePath = modified.FilePath,
                    Status = GitFileStatus.Modified
                };
            }
        }
    }
}