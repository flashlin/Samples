using System.Collections.ObjectModel;

namespace GitCli.Models.ConsoleMixedReality;

public class DropdownListBox : IConsoleElement
{
	private readonly ListBox _listBox;
	private readonly TextBox _textBox;
	private bool _isSelectedMode = false;
	private bool _isSelectMode = false;

	public DropdownListBox(Rect rect)
	{
		DesignRect = rect;
		Children = new StackChildren(this);
		_textBox = new TextBox(Rect.Empty);
		_listBox = new ListBox(Rect.Empty);
	}

	public Color BackgroundColor { get; set; } = ConsoleColor.Blue;
	public StackChildren Children { get; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Color? HighlightBackgroundColor { get; set; }

	public Position CursorPosition
	{
		get
		{
			if (_isSelectMode)
			{
				return _listBox.CursorPosition;
			}

			return _textBox.CursorPosition;
		}
	}

	public Rect DesignRect { get; set; }
	public bool IsTab { get; set; } = true;
	public string Name { get; set; } = string.Empty;
	public IConsoleElement? Parent { get; set; }
	public Rect ViewRect { get; set; }

	public Character this[Position pos]
	{
		get
		{
			if (!ViewRect.Contain(pos))
			{
				return Character.Empty;
			}

			var y = pos.Y - ViewRect.Top;
			if (y == 0)
			{
				return _textBox[pos];
			}

			if (!_isSelectedMode)
			{
				return Character.Empty;
			}

			return _listBox[pos];
		}
	}

	public Rect GetChildrenRect()
	{
		return ViewRect;
	}

	public bool OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
		return false;
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		_textBox.Parent = this;
		_listBox.Parent = this;
		_textBox.ViewRect = new Rect
		{
			Left = rect.Left,
			Top = rect.Top,
			Width = rect.Width,
			Height = 1,
		};
		_listBox.ViewRect = new Rect
		{
			Left = rect.Left,
			Top = rect.Top + 1,
			Width = rect.Width,
			Height = rect.Height - 1,
		};
	}

	public bool OnInput(InputEvent inputEvent)
	{
		if (_isSelectedMode)
		{
			return _listBox.OnInput(inputEvent);
		}

		return _textBox.OnInput(inputEvent);
	}

	public void Refresh()
	{

	}
}