using CommunityToolkit.Mvvm.ComponentModel;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GitMaui.Components
{
	public interface ITreeNode
	{
		public string Title { get; set; }
		public ObservableCollection<TreeNode> Child { get; set; }
	}

	[INotifyPropertyChanged]
	public partial class TreeNode : ITreeNode
	{
		[ObservableProperty]
		string _title;

		[ObservableProperty]
		ObservableCollection<TreeNode> _child = new();
	}

	[INotifyPropertyChanged]
	public partial class TreeNodeCollection
	{
		[ObservableProperty]
		ObservableCollection<TreeNode> _items = new();
	}
}
