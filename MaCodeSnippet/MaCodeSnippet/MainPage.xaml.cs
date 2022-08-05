using CommunityToolkit.Maui.Markup;
using MaCodeSnippet.ViewModels;
using MaCodeSnippet.Views;

namespace MaCodeSnippet;

public partial class MainPage : ContentPage
{
	int count = 0;

	public MainPage(CodeSnippetViewModel mainViewModel)
	{
		InitializeComponent();

		//Content = new CollectionView
		//{
		//	BackgroundColor = Color.FromArgb("F6F6EF"),
		//	SelectionMode = SelectionMode.Single,
		//	ItemTemplate = new MainDataTemplate(),
		//}.Bind(CollectionView.ItemsSourceProperty, nameof(CodeSnippetViewModel.CodeSnippetCollection));
	}
}

