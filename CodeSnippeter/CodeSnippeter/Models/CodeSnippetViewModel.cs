using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using CodeSnippeter.Annotations;

namespace CodeSnippeter.Models
{
	public partial class CodeSnippetViewModel : INotifyObject
	{
		public ObservableCollectionFast<CodeSnippet> CodeSnippets { get; set; } = new();
		public NotifyObject<string> SearchContext;

		private ICodeSnippetService _codeSnippetService;

		public CodeSnippetViewModel(ICodeSnippetService codeSnippetService)
		{
			_codeSnippetService = codeSnippetService;

			SearchContext = new NotifyObject<string>(this, nameof(SearchContext), string.Empty);
			SearchCommand = new ClickCommand(() =>
			{
				var codes = _codeSnippetService.QueryCode(SearchContext.Value);
				CodeSnippets.Clear();
				CodeSnippets.AddRange(codes);
			});
		}

		public ICommand SearchCommand { get; }

		public event PropertyChangedEventHandler? PropertyChanged;

		protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
		{
			PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
		}

		public void RaisePropertyChanged(string propertyName, PropertyChangedEventArgs eventArgs)
		{
			PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
		}

		public ObservableCollectionFast<CodeSnippet> GetCodeSnippetsList()
		{
			var codes = _codeSnippetService.QueryCode(SearchContext.Value);
			CodeSnippets.Clear();
			CodeSnippets.AddRange(codes);
			return CodeSnippets;
		}
	}


	public class ClickCommand : ObservableCommand
	{
		private readonly Action _action;

		public ClickCommand(Action action)
		{
			_action = action;
		}
		protected override void ExecuteCommand(object? parameter)
		{
			_action();
		}
	}
}
