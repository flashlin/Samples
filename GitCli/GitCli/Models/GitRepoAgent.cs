using LibGit2Sharp;

namespace GitCli.Models;

public class GitRepoAgent
{
    public IEnumerable<FileStatusInfo> QueryFileStatus(RepositoryStatus status)
    {
        foreach (var modified in status.Modified)
        {
            yield return new FileStatusInfo
            {
                FilePath = modified.FilePath,
                Status = GitFileStatus.Modified
            };
        }
    }

    public GitRepoInfo OpenRepoFolder()
    {
        var folderPath = "D:/VDisk/Github/Samples";
        using (var repo = new Repository(folderPath))
        {
            //var master = repo.Branches["master"];
            var status = repo.RetrieveStatus();
            //status.IsDirty;
            //status.Modified
            return new GitRepoInfo
            {
                FolderPath = folderPath,
                Status = this.QueryFileStatus(status).ToList()
            };
        }
    }
}