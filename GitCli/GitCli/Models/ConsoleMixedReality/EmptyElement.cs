namespace GitCli.Models.ConsoleMixedReality;

public class EmptyElement : IConsoleEditableElement
{
	public static EmptyElement Default = new();

	private int _editIndex = 0;

	private EmptyElement()
	{
		Children = new StackChildren(this);
	}

	public Color BackgroundColor { get; set; } = ConsoleColor.DarkBlue;
	public StackChildren Children { get; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Color? HighlightBackgroundColor { get; set; }
	public Position CursorPosition => Position.Empty;
	public Rect DesignRect { get; set; } = Rect.Empty;
	public int EditIndex
	{
		get => _editIndex;
		set => _editIndex = 0;
	}

	public bool IsTab { get; set; } = false;
	public string Name { get; set; } = string.Empty;
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

	public bool OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
		return Parent?.OnBubbleEvent(element, inputEvent) ?? false;
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

	public void Refresh()
	{
	}
}