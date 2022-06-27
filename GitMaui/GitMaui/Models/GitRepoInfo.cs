using CommunityToolkit.Mvvm.ComponentModel;
using GitMaui.Components;
using LibGit2Sharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GitMaui.Models
{
	[INotifyPropertyChanged]
	public partial class GitRepoInfo
	{
		public GitRepoInfo(string folderPath)
		{
			FolderPath = folderPath;
		}

		[ObservableProperty]
		string _folderPath;

		[ObservableProperty]
		ObservableCollection<GitFileInfo> _changes = new();

		public IEnumerable<GitFileInfo> QueryStatus()
		{
			using (var repo = new Repository(_folderPath))
			{
				//var master = repo.Branches["master"];
				var status = repo.RetrieveStatus();
				//status.IsDirty;
				foreach (var modified in status.Modified)
				{
					yield return new GitFileInfo
					{
						FilePath = modified.FilePath,
						Status = GitFileStatus.Modified
					};
				}
			}
		}

		public void ConvertToTreeNode(IEnumerable<GitFileInfo> fileList)
		{
		}


		public void ConvertToTreeNode(List<TreeNode> nodeList, GitFileInfo file)
		{
			var ss = file.FilePath.Split(Path.PathSeparator);

		}
	}

	public class GitFileInfo
	{
		public GitFileInfo()
		{
			Title = FilePath;
		}

		string _filePath;
		public string FilePath
		{
			get
			{
				return _filePath;
			}
			set
			{
				if (_filePath != value)
				{
					_filePath = value;
					Title = value;
				}
			}
		}

		public GitFileStatus Status { get; set; }

		public string Title { get; private set; }
	}

	public enum GitFileStatus
	{
		Modified
	}
}
