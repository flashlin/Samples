namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleTextBox : IConsoleElement
{
	private int _editIndex;
	private int _startSelectIndex;

	public ConsoleTextBox(Rect rect)
	{
		GetViewRect = () => rect;
		EditRect = rect;
	}

	public Color Background { get; set; } = ConsoleColor.DarkBlue;
	public IConsoleWriter Console { get; set; }

	public Position CursorPosition
	{
		get
		{
			if (_editIndex < EditRect.Width)
			{
				return new Position(EditRect.Left + _editIndex, EditRect.Top);
			}

			return new Position(EditRect.Left + EditRect.Width, EditRect.Top);
		}
	}

	public int EditIndex => _editIndex;
	public Rect EditRect { get; set; }
	public Func<Rect> GetViewRect { get; set; }
	public bool IsSelectedMode { get; private set; }
	public int MaxLength { get; set; } = int.MaxValue;
	public string Value { get; set; } = String.Empty;

	public Character this[Position pos]
	{
		get
		{
			var rect = EditRect.Intersect(GetViewRect());
			if (!rect.Contain(pos))
			{
				return Character.Empty;
			}

			var contentSpan = GetShowContentSpan(rect);
			var showContent = GetShowContent(contentSpan);

			var x = pos.X - rect.Left;
			var selectedSpan = GetSelectedSpan().Intersect(contentSpan);
			if (!selectedSpan.IsEmpty)
			{
				var selectedValue = GetSelectedValue(selectedSpan);
				if (selectedSpan.Contain(x))
				{
					return new Character(selectedValue[x - selectedSpan.Index], null, Color.DarkGray);
				}
			}

			if (x >= showContent.Length)
			{
				return new Character(' ', null, Background);
			}

			return new Character(showContent[x], null, Background);
		}
	}

	public bool OnInput(InputEvent inputEvent)
	{
		var newText = (string?)null;

		if (!IsSelectedMode && inputEvent.HasShift)
		{
			_startSelectIndex = _editIndex;
			IsSelectedMode = true;
		}

		switch (inputEvent.Key)
		{
			case ConsoleKey.LeftArrow:
				_editIndex = Math.Max(0, _editIndex - 1);
				IsSelectedMode = (inputEvent.HasShift);
				break;

			case ConsoleKey.RightArrow:
				_editIndex = Math.Min(Value.Length, _editIndex + 1);
				IsSelectedMode = (inputEvent.HasShift);
				break;

			case ConsoleKey.Backspace:
				_editIndex = Math.Max(0, _editIndex - 1);
				newText = $"{Value.Substring(0, _editIndex)}{Value.SubStr(_editIndex + 1)}";
				IsSelectedMode = false;
				break;

			case ConsoleKey.Delete:
				if (IsSelectedMode)
				{
					var showContentSpan = GetShowContentSpanByView();
					var selectedSpan = GetSelectedSpan();
					var remainingSpans = selectedSpan.NonIntersect(showContentSpan).ToArray();
					newText = string.Join("", remainingSpans.Select(x => Value.Substring(x.Index, x.Length)));
				}
				else
				{
					newText = $"{Value.Substring(0, _editIndex)}{Value.SubStr(_editIndex + 1)}";
				}
				IsSelectedMode = false;
				break;

			case ConsoleKey.Home:
				_editIndex = 0;
				break;

			case ConsoleKey.End:
				_editIndex = Value.Length;
				break;

			default:
				if (Value.Length + 1 > MaxLength)
				{
					break;
				}
				var character = inputEvent.Key == ConsoleKey.Enter
					 ? '\n'
					 : inputEvent.KeyChar;
				newText = $"{Value.Substring(0, _editIndex)}{character}{Value.SubStr(_editIndex)}";
				_editIndex = Math.Min(newText.Length, _editIndex + 1);
				IsSelectedMode = false;
				break;
		}

		if (newText != null)
		{
			Value = newText;
		}

		return true;
	}

	private StrSpan GetSelectedSpan()
	{
		if (!IsSelectedMode)
		{
			return StrSpan.Empty;
		}

		var startIndex = Math.Min(_editIndex, _startSelectIndex);
		var endIndex = Math.Max(_editIndex, _startSelectIndex);
		return new StrSpan
		{
			Index = startIndex,
			Length = endIndex - startIndex,
		};
	}

	private string GetSelectedValue(StrSpan selectedSpan)
	{
		return Value.Substring(selectedSpan.Index, selectedSpan.Length);
	}

	private string GetShowContent(StrSpan contentSpan)
	{
		return Value.Substring(contentSpan.Index, contentSpan.Length);
	}

	private StrSpan GetShowContentSpanByView()
	{
		var rect = EditRect.Intersect(GetViewRect());
		return GetShowContentSpan(rect);
	}

	private StrSpan GetShowContentSpan(Rect rect)
	{
		var startIndex = _editIndex - rect.Width;
		if (_editIndex < rect.Width)
		{
			startIndex = 0;
		}

		var len = Math.Min(Value.Length, rect.Width);
		return new StrSpan
		{
			Index = startIndex,
			Length = len
		};
	}
}