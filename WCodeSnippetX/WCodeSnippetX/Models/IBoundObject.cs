namespace WCodeSnippetX.Models;

public interface IBoundObject
{
	string QueryCodeAsync(string text);
}