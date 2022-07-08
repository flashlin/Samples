using System.ComponentModel;
using LibGit2Sharp;

namespace GitCli.Models;

public class GitRepoInfo : IObjectNotifyPropertyChanged
{
	public GitRepoInfo()
	{
      Status = new NotifyProperty<IEnumerable<FileStatusInfo>>(this, nameof(Status), Enumerable.Empty<FileStatusInfo>());
	}

    public string FolderPath { get; init; } = String.Empty;
    
    public NotifyProperty<IEnumerable<FileStatusInfo>> Status { get; init; }

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

    public event PropertyChangedEventHandler? PropertyChanged;
    public void RaisePropertyChanged(string propertyName)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    public IEnumerable<GitBranchInfo> QueryBranches()
    {
        var branches = Repository.ListRemoteReferences(FolderPath);
        var refHeads = "refs/heads/";
        foreach(var branch in branches)
        {
            if (branch.TargetIdentifier.StartsWith(refHeads) || 
                branch.CanonicalName.StartsWith(refHeads))
            {
                yield return new GitBranchInfo
                {
                    Name = branch.CanonicalName.Replace(refHeads, String.Empty),
                    IsLocalBranch = true,
                    TargetIdentifier = branch.TargetIdentifier
                };
            }
            else
            {
                yield return new GitBranchInfo
                {
                    Name = branch.CanonicalName.Replace("refs/remotes/", String.Empty),
                    IsLocalBranch = false,
                    TargetIdentifier = branch.TargetIdentifier
                };
            }
        }
    }
}

public class GitBranchInfo
{
    public string Name { get; set; }
    public bool IsLocalBranch { get; set; }
    public string TargetIdentifier { get; set; }
}