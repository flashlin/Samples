namespace GitCli.Models.ConsoleMixedReality;

public class ListBox : IConsoleElement
{
	private int _editIndex;
	private int _index = -1;
	private int _startSelectIndex;
	private bool _isSelectedMode;
	private int _maxLength;

	public ListBox(Rect rect)
	{
		ViewRect = rect;
	}

	public IConsoleElement? Parent { get; set; }

	public List<TextBox> Children { get; set; } = new List<TextBox>();

	public Color BackgroundColor { get; set; } = ConsoleColor.Blue;

	public Position CursorPosition
	{
		get
		{
			if (_index >= 0)
			{
				return Children[_index].CursorPosition;
			}
			return new Position(ViewRect.Left, ViewRect.Top);
		}
	}

	public Rect ViewRect { get; set; }
	public int MaxLength { get; set; } = int.MaxValue;

	public Character this[Position pos]
	{
		get
		{
			if (!ViewRect.Contain(pos))
			{
				return Character.Empty;
			}

			if (Children.Count <= 0)
			{
				return new Character(' ', null, BackgroundColor);
			}

			var x = pos.X - ViewRect.Left;
			var y = _index + pos.Y - ViewRect.Top;
			if (_index < ViewRect.Height)
			{
				y = pos.Y - ViewRect.Top;
			}

			if (y < Children.Count)
			{
				return Children[y][pos];
			}

			return new Character(' ', null, BackgroundColor);
		}
	}

	public bool OnInput(InputEvent inputEvent)
	{
		var prevEditIndex = 0;
		switch (inputEvent.Key)
		{
			case ConsoleKey.LeftArrow:
				//_editIndex = Math.Max(0, _editIndex - 1);
				GetFocusedListItem().OnInput(inputEvent);
				break;

			case ConsoleKey.RightArrow:
				//if (_editIndex + 1 > _maxLength)
				//{
				//	break;
				//}
				//_editIndex = Math.Min(_maxLength, _editIndex + 1);
				GetFocusedListItem().OnInput(inputEvent);
				break;

			case ConsoleKey.UpArrow:
				if (_index == -1)
				{
					break;
				}
				prevEditIndex = GetFocusedListItem().EditIndex;
				_index = Math.Max(_index - 1, 0);
				GetFocusedListItem().EditIndex = prevEditIndex;
				break;
			case ConsoleKey.DownArrow:
				if (_index == -1)
				{
					break;
				}
				prevEditIndex = GetFocusedListItem().EditIndex;
				_index = Math.Min(_index + 1, Children.Count - 1);
				GetFocusedListItem().EditIndex = prevEditIndex;
				break;

			// case ConsoleKey.Home:
			//     _editIndex = 0;
			//     break;
			//
			// case ConsoleKey.End:
			//     _editIndex = Value.Length;
			//     break;

			case ConsoleKey.Enter:
				break;
		}

		return true;
	}

	public void OnCreated(IConsoleWriter console)
	{
		var y = ViewRect.Top;
		foreach (var child in Children)
		{
			_index = 0;
			child.Parent = this;
			child.ViewRect = new Rect()
			{
				Left = ViewRect.Left,
				Top = y,
				Width = ViewRect.Width,
				Height = 1,
			};
			child.OnCreated(console);
			_maxLength = Math.Max(_maxLength, child.Value.Length);
			y += 1;
		}
	}

	public void OnBubbleEvent(InputEvent inputEvent)
	{
	}

	private IConsoleEditableElement GetFocusedListItem()
	{
		if (_index == -1)
		{
			return new EmptyElement();
		}
		return Children[_index];
	}
}