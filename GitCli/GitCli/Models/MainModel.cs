using GitCli.Models.ConsoleMixedReality;

namespace GitCli.Models;

public class MainModel
{
	public MainModel()
	{
		LocalChangesCommand = new EntryCommand(("All Commits", OnHandleAllChanges));
	}

	public GitRepoInfo RepoInfo { get; set; }
	public NotifyCollection<ListItem> ChangesList { get; set; } = new();
	public NotifyCollection<ListItem> BranchList { get; set; } = new();
	public NotifyCollection<ListItem> AllCommitList { get; set; } = new();
	public NotifyCollection<ListItem> CompareList { get; set; } = new();
	public NotifyCollection<ListItem> ChangedFilesList { get; set; } = new();
	public IModelCommand LocalChangesCommand { get; set; }

	private void OnHandleAllChanges()
	{
		var commits = RepoInfo.QueryCommits();
		foreach (var commit in commits)
		{
			AllCommitList.Adding(new ListItem()
			{
				Title = commit.Message,
				Value = commit
			});
		}
		AllCommitList.Notify();
	}

	private void OnHandleACommit(IConsoleElement target)
	{
		var commits = RepoInfo.QueryCommits();
		foreach (var commit in commits)
		{
			AllCommitList.Adding(new ListItem()
			{
				Title = commit.Message,
				Value = commit
			});
		}
		AllCommitList.Notify();
	}
}