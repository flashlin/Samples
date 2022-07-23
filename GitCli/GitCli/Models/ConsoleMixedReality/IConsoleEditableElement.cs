namespace GitCli.Models.ConsoleMixedReality;

public interface IConsoleEditableElement : IConsoleElement
{
	int EditIndex { get; set; }
	string Value { get; }
	void ForceSetEditIndex(int index);
}