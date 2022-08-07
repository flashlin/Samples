namespace WCodeSnippetX.Models;

public interface ICodeSnippetRepo
{
	IEnumerable<CodeSnippetEntity> QueryCode(string text);
	void UpdateCode(CodeSnippetEntity codeSnippet);
	void AddCode(CodeSnippetEntity codeSnippet);
	void DeleteCode(CodeSnippetEntity codeSnippet);
}