namespace GitCli.Models.ConsoleMixedReality;

public class EmptyElement : IConsoleEditableElement
{
	private int _editIndex = 0;
	public Position CursorPosition => Position.Empty;
	public int EditIndex
	{
		get => _editIndex;
		set => _editIndex = 0;
	}

	public IConsoleElement? Parent { get; set; }
	public string Value { get; } = String.Empty;
	public void ForceSetEditIndex(int index)
	{
		_editIndex = index;
	}

	public Rect ViewRect { get; set; } = Rect.Empty;
	public Character this[Position pos] => Character.Empty;

	public void OnBubbleEvent(InputEvent inputEvent)
	{
		Parent?.OnBubbleEvent(inputEvent);
	}

	public void OnCreate(IConsoleManager manager)
	{
	}

	public bool OnInput(InputEvent inputEvent)
	{
		return false;
	}
}