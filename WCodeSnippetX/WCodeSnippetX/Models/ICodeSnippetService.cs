namespace WCodeSnippetX.Models;

public interface ICodeSnippetService
{
	List<CodeSnippetEntity> Query(string text);
	void AddCode(CodeSnippetEntity code);
	void UpdateCode(CodeSnippetEntity code);
	void DeleteCodeById(int id);
}