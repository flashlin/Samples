using GitCli.Models.ConsoleMixedReality;

namespace GitCli.Models;

public class MainModel
{
	public MainModel()
	{
		LocalChangesCommand = new EntryCommand("All Commits", OnHandleAllChanges);
		ACommitCommand = new ExecuteCommand(OnHandleACommit);
	}

	public GitRepoInfo RepoInfo { get; set; }
	public NotifyCollection<ListItem> ChangesList { get; set; } = new();
	public NotifyCollection<ListItem> BranchList { get; set; } = new();
	public NotifyCollection<ListItem> AllCommitList { get; set; } = new();
	public NotifyCollection<ListItem> CompareList { get; set; } = new();
	public NotifyCollection<ListItem> ChangedFilesList { get; set; } = new();
	public IModelCommand LocalChangesCommand { get; }
	public IModelCommand ACommitCommand { get; }

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

	private void OnHandleACommit(ConsoleElementEvent evt)
	{
		var commit = (GitCommitInfo)evt.Element.UserObject;
		var count = 0;
		foreach (var entry in commit.Tree)
		{
			count++;
		}
	}
}