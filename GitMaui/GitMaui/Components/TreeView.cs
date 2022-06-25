using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GitMaui.Components
{
	public class TreeView : ContentView
	{
		public TreeView()
		{
			BindingContext = new List<TreeNode>
			{
				new TreeNode
				{
					Title = "Title1",
					Child = new ObservableCollection<TreeNode>
					{
						new TreeNode { Title = "A1" },
						new TreeNode { Title = "A2" },
						new TreeNode { Title = "A3" },
					}
				},
				new TreeNode
				{
					Title = "Title2",
					Child = new ObservableCollection<TreeNode>
					{
						new TreeNode { Title = "B1" },
						new TreeNode { Title = "B2" },
						new TreeNode { Title = "B3" },
					}
				},
				new TreeNode
				{
					Title = "Title3",
					Child = new ObservableCollection<TreeNode>
					{
						new TreeNode { Title = "C1" },
						new TreeNode { Title = "C2" },
						new TreeNode { Title = "C3" },
					}
				},
			};

			Content = new CollectionView
			{
				ItemTemplate = new DataTemplate
				{

				}
			};
		}
	}
}
