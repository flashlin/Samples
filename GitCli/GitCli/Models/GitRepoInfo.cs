using System.ComponentModel;
using LibGit2Sharp;

namespace GitCli.Models;

public class GitRepoInfo : IObjectNotifyPropertyChanged
{
	public GitRepoInfo()
	{
      Status = new NotifyProperty<IEnumerable<FileStatusInfo>>(this, nameof(Status), Enumerable.Empty<FileStatusInfo>());
	}

    public string FolderPath { get; init; }
    
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
}