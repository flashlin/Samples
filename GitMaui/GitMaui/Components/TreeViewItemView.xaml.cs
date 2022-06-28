using GitMaui.Helpers;

namespace GitMaui.Components;

public partial class TreeViewItemView : ContentView
{
	public TreeViewItemView()
	{
		InitializeComponent();
		SetIconImage();

		BindingContextChanged += TreeViewItemView_BindingContextChanged;
	}

	private void TreeViewItemView_BindingContextChanged(object sender, EventArgs e)
	{
		var model = BindingContext as IHasChildrenTreeViewNode;
		extended.IsVisible = model.IsExtended;

		if (model.IsExtended)
		{
			//extendButton.RotateTo(0);
			if (model is ILazyLoadTreeViewNode lazyNode && lazyNode.GetChildren != null && !lazyNode.Children.Any())
			{
				foreach (var child in lazyNode.GetChildren(lazyNode))
				{
					lazyNode.Children.Add(child);
					//extended.Add(new TreeViewNodeView(child, itemTemplate));
					var subTreeItem = new TreeViewItemView();
					extended.Add(subTreeItem);
				}

				if (!lazyNode.Children.Any())
				{
					extended.Opacity = 0;
					lazyNode.IsLeaf = true;
				}
			}
		}
	}

	//public static readonly BindableProperty SourceProperty =
	//	BindableProperty.Create(nameof(Source),
	//		typeof(IHasChildrenTreeViewNode),
	//		typeof(TreeViewItemView),
	//		propertyChanged: (b, o, v) => (b as TreeViewItemView).OnSourceSet());

	//public IHasChildrenTreeViewNode Source
	//{
	//	get => (IHasChildrenTreeViewNode)GetValue(SourceProperty);
	//	set => SetValue(SourceProperty, value);
	//}

	//protected virtual void OnSourceSet()
	//{
	//	//root.ItemsSource = ItemsSource;
	//}

	public void SetIconImage()
	{
		//icon.Source = new FontImageSource
		//{
		//	Glyph = IconFont.SquarePlus,
		//	FontFamily = DeviceInfo.Platform == DevicePlatform.iOS ? "Ionicons" : "FontAwesome",
		//	Size = 16,
		//	Color = Color.Parse("Black")
		//};
		//(icon.Source as FontImageSource).Glyph = IconFont.SquarePlus;
		font.Glyph = IconFont.SquareMinus;
	}
}