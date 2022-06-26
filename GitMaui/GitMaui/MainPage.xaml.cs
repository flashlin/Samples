using GitMaui.Models;
using GitMaui.ViewModels;
using Microsoft.Maui.Controls;

namespace GitMaui;

public partial class MainPage : ContentPage
{

	public MainPage(MainViewModel vm)
	{
		InitializeComponent();
		BindingContext = vm;

		vm.GitRepoInfo = new GitRepoInfo("D:/VDisk/Github/Samples")
		{
		};

		vm.GitRepoInfo.Changes.Clear();
		foreach (var item in vm.GitRepoInfo.QueryStatus())
		{
			vm.GitRepoInfo.Changes.Add(item);
		}

		//Content.SetBinding(VerticalStackLayout.BindingContextProperty, nameof(vm.GitRepoInfo), BindingMode.TwoWay);
		//tree1.Content.SetBinding(VerticalStackLayout.BindingContextProperty,
		//	new Binding(nameof(vm.GitRepoInfo.Changes))
		//	{
		//		Source = vm.GitRepoInfo,
		//	});
	}

	private void OnCounterClicked(object sender, EventArgs e)
	{
		//if (count == 1)
		//	CounterBtn.Text = $"Clicked {count} time";
		//else
		//	CounterBtn.Text = $"Clicked {count} times";

		//SemanticScreenReader.Announce(CounterBtn.Text);

		//label.SetBinding(Label.TextProperty, "MyName");

		//var entry = new Entry();
		//entry.SetBinding<GitFileInfo>(Entry.TextProperty, vm => vm.FilePath, BindingBase.EnableCollectionSynchronization);
	}
}

