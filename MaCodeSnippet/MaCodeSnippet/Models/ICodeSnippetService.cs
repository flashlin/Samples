using System.Collections.Generic;

namespace MaCodeSnippet.Models;

public interface ICodeSnippetService
{
	IEnumerable<CodeSnippet> QueryCode(string text);
}