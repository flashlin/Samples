using LibGit2Sharp;

namespace GitCli.Models;

public class GitRepoAgent : IGitRepoAgent
{
    public GitRepoInfo OpenRepoFolder(string folderPath)
    {
        return new GitRepoInfo()
        {
            FolderPath = folderPath
        };
    }
}