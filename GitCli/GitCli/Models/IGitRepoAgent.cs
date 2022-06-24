namespace GitCli.Models;

public interface IGitRepoAgent
{
    GitRepoInfo OpenRepoFolder(string folderPath);
}