using System.Collections.Generic;

namespace CodeSnippeter.Models;

public interface ICodeSnippetService
{
	IEnumerable<CodeSnippet> QueryCode(string text);
}