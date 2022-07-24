namespace GitCli.Models.ConsoleMixedReality;

public interface IConsoleEditableElement : IConsoleElement
{
	int EditIndex { get; set; }
	void ForceSetEditIndex(int index);
}