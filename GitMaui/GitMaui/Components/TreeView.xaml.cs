using System.Collections;
using System.Collections.Specialized;

namespace GitMaui.Components;

public partial class TreeView : ContentView
{
	public TreeView()
	{
		InitializeComponent();
	}

	public static readonly BindableProperty ItemsSourceProperty =
		BindableProperty.Create(nameof(ItemsSource),
			typeof(IEnumerable),
			typeof(TreeView1),
			null,
			propertyChanging: (b, o, n) => (b as TreeView).OnItemsSourceSetting(o as IEnumerable, n as IEnumerable),
			propertyChanged: (b, o, v) => (b as TreeView).OnItemsSourceSet());

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
		root.ItemsSource = ItemsSource;
		//Render();
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
		//Render();
	}

	void Render()
	{
		foreach (var item in ItemsSource)
		{
			if (item is IHasChildrenTreeViewNode node)
			{
				//children.Add(new TreeViewNodeView(node, ItemTemplate));
			}
		}
	}
}