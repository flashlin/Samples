using CommunityToolkit.Mvvm.ComponentModel;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using T1.Standard.Extensions;

namespace GitMaui.Components
{
	public interface ITreeNode
	{
		string Title { get; set; }
		bool IsExtended { get; set; }
		object Tag { get; set; }
	}

	public interface IHasChildrenTreeViewNode : ITreeNode
	{
		ObservableCollection<IHasChildrenTreeViewNode> Children { get; }
		bool IsLeaf { get; set; }
	}

	public interface ILazyLoadTreeViewNode : IHasChildrenTreeViewNode
	{
		Func<ITreeNode, IEnumerable<IHasChildrenTreeViewNode>> GetChildren { get; }
	}

	[INotifyPropertyChanged]
	public partial class TreeNode : IHasChildrenTreeViewNode
	{
		[ObservableProperty]
		string _title;

		[ObservableProperty]
		ObservableCollection<TreeNode> _child = new();

		public ObservableCollection<IHasChildrenTreeViewNode> Children
		{
			get
			{
				return new ObservableCollection<IHasChildrenTreeViewNode>(_child);
			}
		}

		[ObservableProperty]
		bool _isExtended;

		[ObservableProperty]
		bool _isLeaf;

		public object Tag { get; set; }

		public override string ToString()
		{
			return $"{Title}-{Tag}";
		}
	}

	[INotifyPropertyChanged]
	public partial class TreeNodeCollection
	{
		[ObservableProperty]
		ObservableCollection<TreeNode> _items = new();
	}
}
