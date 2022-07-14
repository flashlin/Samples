namespace GitCli.Models.ConsoleMixedReality;

public class EmptyElement : IConsoleEditableElement
{
	private int _editIndex = 0;
	public Position CursorPosition => Position.Empty;
	public Rect DesignRect { get; set; } = Rect.Empty;

	public int EditIndex
	{
		get => _editIndex;
		set => _editIndex = 0;
	}

	public bool IsTab { get; set; } = false;
	public IConsoleElement? Parent { get; set; }
	public string Value { get; } = String.Empty;
	public Rect ViewRect { get; set; } = Rect.Empty;

	public Character this[Position pos] => Character.Empty;

	public void ForceSetEditIndex(int index)
	{
		_editIndex = index;
	}
	public Rect GetChildrenRect()
	{
		return Rect.Empty;
	}

	public void Refresh()
	{
	}

	public void OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
		Parent?.OnBubbleEvent(element, inputEvent);
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
	}

	public bool OnInput(InputEvent inputEvent)
	{
		switch (inputEvent.Key)
		{
			case ConsoleKey.Tab:
				OnBubbleEvent(this, inputEvent);
				return false;
		}
		return false;
	}
}