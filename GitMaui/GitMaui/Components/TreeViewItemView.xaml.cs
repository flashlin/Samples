using GitMaui.Helpers;

namespace GitMaui.Components;

public partial class TreeViewItemView : ContentView
{
	public TreeViewItemView()
	{
		InitializeComponent();
		SetIconImage();
	}

	//Glyph="{x:Static helpers:IconFont.SquarePlus}"
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