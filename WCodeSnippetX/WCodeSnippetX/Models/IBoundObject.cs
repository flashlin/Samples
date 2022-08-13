namespace WCodeSnippetX.Models;

public interface IBoundObject
{
	string QueryCode(string text);
	FormMainCef Form { get; set; }
	void Minimize();
	void BringMeToFront();
	void SetClipboard(string text);
}