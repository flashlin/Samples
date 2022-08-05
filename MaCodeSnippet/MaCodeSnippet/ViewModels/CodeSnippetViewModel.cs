using System;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MaCodeSnippet.Models;

namespace MaCodeSnippet.ViewModels
{
	[INotifyPropertyChanged]
	public partial class CodeSnippetViewModel
	{
		readonly WeakEventManager _pullToRefreshEventManager = new();
		private ICodeSnippetService _codeSnippetService;

		public CodeSnippetViewModel(ICodeSnippetService codeSnippetService)
		{
			_codeSnippetService = codeSnippetService;
		}


		public ObservableCollectionFast<CodeSnippet> CodeSnippetCollection { get; set; } = new();

		[ObservableProperty]
		private string _searchContext = String.Empty;

		public event EventHandler<string> PullToRefreshFailed
		{
			add => _pullToRefreshEventManager.AddEventHandler(value);
			remove => _pullToRefreshEventManager.RemoveEventHandler(value);
		}

		[ICommand]
		public Task SearchCommand()
		{
			var codes = _codeSnippetService.QueryCode(_searchContext);
			CodeSnippetCollection.Clear();
			CodeSnippetCollection.AddRange(codes);
			return Task.CompletedTask;
		}
	}
}
