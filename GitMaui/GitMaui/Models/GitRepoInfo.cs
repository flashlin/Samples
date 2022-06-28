using CommunityToolkit.Mvvm.ComponentModel;
using GitMaui.Components;
using LibGit2Sharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using T1.Standard.Collections;

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

		[ObservableProperty]
		ObservableCollectionFast<TreeNode> _changesTree = new();

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

		public List<TreeNode> ConvertToTreeNode(IEnumerable<GitFileInfo> fileList)
		{
			var treeBuilder = new TreeBuilder();
			var treeList = treeBuilder.ReduceTree(fileList,
				x => x.FilePath,
				x => treeBuilder.GetParentPath(x),
				x => treeBuilder.QueryParentPaths(x.FilePath)).ToList();

			return ConvertTree(treeList).ToList();
		}

		private IEnumerable<TreeNode> ConvertTree(IEnumerable<TreeBuilder.TreeItem<GitFileInfo, string>> treeList)
		{
			foreach(var treeItem in treeList)
			{
				yield return ToTreeNode(treeItem);
			}
		}

		private TreeNode ToTreeNode(TreeBuilder.TreeItem<GitFileInfo, string> node)
		{
			if( node.Item == null)
			{
				return new TreeNode
				{
					Title = node.Id,
					Child = new ObservableCollection<TreeNode>(node.Children.Select(x => ToTreeNode(x)))
				};
			}

			return new TreeNode
			{
				Title = node.Id,
				Tag = node.Item,
				IsLeaf = true,
			};
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
