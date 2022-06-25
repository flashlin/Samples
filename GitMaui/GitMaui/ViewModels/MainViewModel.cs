using GitMaui.Models;
using CommunityToolkit.Mvvm.ComponentModel;

namespace GitMaui.ViewModels
{
	[INotifyPropertyChanged]
	public partial class MainViewModel
	{
		[ObservableProperty]
		GitRepoInfo _gitRepoInfo;
	}
}
