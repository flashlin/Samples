using System.Collections.Generic;
using System.Linq;

namespace MaCodeSnippet.Models;

public class CodeSnippetService : ICodeSnippetService
{
	public IEnumerable<CodeSnippet> QueryCode(string text)
	{
		return CodeSnippets().Where(x => x.Context.Contains(text));
	}

	private IEnumerable<CodeSnippet> CodeSnippets()
	{
		yield return new CodeSnippet
		{
			Context = "a",
		};
		yield return new CodeSnippet
		{
			Context = "b",
		};
		yield return new CodeSnippet
		{
			Context = "c",
		};
	}
}