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
		string Title { get; set; }
		bool IsExtended { get; set; }
	}

	public interface IHasChildrenTreeViewNode : ITreeNode
	{
		IList<IHasChildrenTreeViewNode> Children { get; }
		bool IsLeaf { get; set; }
	}

	public interface ILazyLoadTreeViewNode : IHasChildrenTreeViewNode
	{
		Func<ITreeNode, IEnumerable<IHasChildrenTreeViewNode>> GetChildren { get; }
	}

	[INotifyPropertyChanged]
	public partial class TreeNode : ITreeNode
	{
		[ObservableProperty]
		string _title;

		[ObservableProperty]
		ObservableCollection<TreeNode> _child = new();

		[ObservableProperty]
		bool _isExtended;

		[ObservableProperty]
		bool _isLeaf;
	}

	[INotifyPropertyChanged]
	public partial class TreeNodeCollection
	{
		[ObservableProperty]
		ObservableCollection<TreeNode> _items = new();
	}
}
