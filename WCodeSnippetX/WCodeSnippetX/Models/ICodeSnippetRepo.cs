namespace WCodeSnippetX.Models;

public interface ICodeSnippetRepo
{
	IEnumerable<CodeSnippetEntity> QueryCode(string text);
}