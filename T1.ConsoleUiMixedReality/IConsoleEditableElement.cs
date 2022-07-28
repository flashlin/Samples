namespace T1.ConsoleUiMixedReality;

public interface IConsoleEditableElement : IConsoleElement
{
	int EditIndex { get; set; }
	void ForceSetEditIndex(int index);
}