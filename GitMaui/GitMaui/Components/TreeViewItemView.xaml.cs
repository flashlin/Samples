using GitMaui.Helpers;

namespace GitMaui.Components;

public partial class TreeViewItemView : ContentView
{
	public TreeViewItemView()
	{
		InitializeComponent();
		SetIconImage();
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