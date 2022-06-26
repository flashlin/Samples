using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GitMaui.Components
{
	public class TreeView1 : ContentView
	{
		private CollectionView _root;

		public TreeView1()
		{
			//BindingContext = new List<ITreeNode>
			//{
			//	new TreeNode
			//	{
			//		Title = "Title1",
			//		Child = new ObservableCollection<TreeNode>
			//		{
			//			new TreeNode { Title = "A1" },
			//			new TreeNode { Title = "A2" },
			//			new TreeNode { Title = "A3" },
			//		}
			//	},
			//	new TreeNode
			//	{
			//		Title = "Title2",
			//		Child = new ObservableCollection<TreeNode>
			//		{
			//			new TreeNode { Title = "B1" },
			//			new TreeNode { Title = "B2" },
			//			new TreeNode { Title = "B3" },
			//		}
			//	},
			//	new TreeNode
			//	{
			//		Title = "Title3",
			//		Child = new ObservableCollection<TreeNode>
			//		{
			//			new TreeNode { Title = "C1" },
			//			new TreeNode { Title = "C2" },
			//			new TreeNode { Title = "C3" },
			//		}
			//	},
			//};

			Content = _root = new CollectionView();
		}

		public static readonly BindableProperty ItemsSourceProperty =
			BindableProperty.Create(nameof(ItemsSource),
				typeof(IEnumerable),
				typeof(TreeView1),
				null,
				propertyChanging: (b, o, n) => (b as TreeView1).OnItemsSourceSetting(o as IEnumerable, n as IEnumerable),
				propertyChanged: (b, o, v) => (b as TreeView1).OnItemsSourceSet());

		public static readonly BindableProperty ItemTemplateProperty =
			BindableProperty.Create(nameof(ItemTemplate),
				typeof(DataTemplate),
				typeof(TreeView1),
				new DataTemplate(typeof(DefaultTreeViewNodeView)),
				propertyChanged: (b, o, n) => (b as TreeView1).OnItemTemplateChanged());

		public DataTemplate ItemTemplate
		{
			get => (DataTemplate)GetValue(ItemTemplateProperty);
			set => SetValue(ItemTemplateProperty, value);
		}

		public IEnumerable ItemsSource
		{
			get => (IEnumerable)GetValue(ItemsSourceProperty);
			set => SetValue(ItemsSourceProperty, value);
		}

		protected virtual void OnItemsSourceSetting(IEnumerable oldValue, IEnumerable newValue)
		{
			if (oldValue is INotifyCollectionChanged oldItemsSource)
			{
				oldItemsSource.CollectionChanged -= Observable_CollectionChanged;
			}

			if (newValue is INotifyCollectionChanged newItemsSource)
			{
				newItemsSource.CollectionChanged += Observable_CollectionChanged;
			}
		}

		protected virtual void OnItemsSourceSet()
		{
			Render();
		}

		private void Observable_CollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
		{
			switch (e.Action)
			{
				case NotifyCollectionChangedAction.Add:
					break;
				case NotifyCollectionChangedAction.Remove:
					break;
				case NotifyCollectionChangedAction.Replace:
					break;
				case NotifyCollectionChangedAction.Move:
					break;
				case NotifyCollectionChangedAction.Reset:
					break;
				default:
					break;
			}
			Render();
		}

		protected virtual void OnItemTemplateChanged()
		{
			Render();
		}

		void Render()
		{
			_root.ItemTemplate = new DataTemplate
			{
				LoadTemplate = () => Create.VerticalStackLayout(children =>
				{
					foreach (var item in ItemsSource)
					{
						if (item is IHasChildrenTreeViewNode node)
						{
							children.Add(new TreeViewNodeView(node, ItemTemplate));
						}
					}
				})
			};
			_root.ItemTemplate.CreateContent();
		}
	}

	public static class Create
	{
		public static VerticalStackLayout VerticalStackLayout(Action<VerticalStackLayout> action)
		{
			var layout = new VerticalStackLayout();
			//layout.SetBinding(ItemsView.ItemsSourceProperty, ".");
			action(layout);
			return layout;
		}
	}

	public class TreeViewNodeView : ContentView
	{
		public TreeViewNodeView(IHasChildrenTreeViewNode node, DataTemplate itemTemplate)
		{
			var sl = new StackLayout { Spacing = 0 };
			Content = sl;

			var slChildrens = new StackLayout 
			{ 
				IsVisible = node.IsExtended, 
				Margin = new Thickness(10, 0, 0, 0), 
				Spacing = 0 
			};

			var extendButton = new ImageButton
			{
				Source = Application.Current.RequestedTheme == AppTheme.Dark ? "down_light.png" : "down_dark.png",
				VerticalOptions = LayoutOptions.Center,
				Opacity = node.IsLeaf ? 0 : 1,
				Rotation = node.IsExtended ? 0 : -90,
				HeightRequest = 30,
				WidthRequest = 30,
			};

			extendButton.Clicked += (s, e) =>
			{
				node.IsExtended = !node.IsExtended;
				slChildrens.IsVisible = node.IsExtended;

				if (node.IsExtended)
				{
					extendButton.RotateTo(0);

					if (node is ILazyLoadTreeViewNode lazyNode && lazyNode.GetChildren != null && !lazyNode.Children.Any())
					{
						foreach (var child in lazyNode.GetChildren(lazyNode))
						{
							lazyNode.Children.Add(child);
							slChildrens.Add(new TreeViewNodeView(child, itemTemplate));
						}

						if (!lazyNode.Children.Any())
						{
							extendButton.Opacity = 0;
							lazyNode.IsLeaf = true;
						}
					}
				}
				else
				{
					extendButton.RotateTo(-90);
				}
			};

			var content = itemTemplate.CreateContent() as View;
			content.BindingContext = node;

			sl.Children.Add(new StackLayout
			{
				Orientation = StackOrientation.Horizontal,
				Children =
				{
					extendButton,
					content
				}
			});

			foreach (var child in node.Children)
			{
				slChildrens.Children.Add(new TreeViewNodeView(child, itemTemplate));
			}

			sl.Children.Add(slChildrens);
		}
	}
}
