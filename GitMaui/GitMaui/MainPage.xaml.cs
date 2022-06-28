using GitMaui.Models;
using GitMaui.ViewModels;
using Microsoft.Maui.Controls;
using System.Collections.Specialized;

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

		var files = vm.GitRepoInfo.QueryStatus().ToArray();
		
		vm.GitRepoInfo.ChangesTree.AddRange(vm.GitRepoInfo.ConvertToTreeNode(files));

		vm.GitRepoInfo.Changes.Clear();
		foreach (var item in files)
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



public class ObservableCollectionFast<T> : ObservableCollection<T>
{
	public ObservableCollectionFast() : base() { }

	public ObservableCollectionFast(IEnumerable<T> collection) : base(collection) { }

	public void AddRange(IEnumerable<T> range)
	{
		foreach (var item in range)
		{
			Items.Add(item);
		}
		this.OnPropertyChanged(new PropertyChangedEventArgs("Count"));
		this.OnPropertyChanged(new PropertyChangedEventArgs("Item[]"));
		this.OnCollectionChanged(new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Reset));
	}

	public void Reset(IEnumerable<T> range)
	{
		this.Items.Clear();
		AddRange(range);
	}
}