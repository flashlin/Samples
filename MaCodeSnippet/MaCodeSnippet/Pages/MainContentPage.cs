using AndroidX.Lifecycle;
using CommunityToolkit.Maui.Markup;
using MaCodeSnippet.ViewModels;
using MaCodeSnippet.Views;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static CommunityToolkit.Maui.Markup.GridRowsColumns;

namespace MaCodeSnippet.Pages
{
	public class MainContentPage : ContentPage
	{
		public MainContentPage()
		{
			Content = new CollectionView
			{
				BackgroundColor = Color.FromArgb("F6F6EF"),
				SelectionMode = SelectionMode.Single,
				ItemTemplate = new MainDataTemplate(),
			}.Bind(CollectionView.ItemsSourceProperty, nameof(CodeSnippetViewModel.CodeSnippetCollection));
		}
	}
}
