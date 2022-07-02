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
	public bool IsSelectedMode { get; set; }
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
		var rect = EditRect.Intersect(GetViewRect());
		var newText = (string?)null;

		if (!IsSelectedMode && inputEvent.HasShift)
		{
			_startSelectIndex = _editIndex;
			IsSelectedMode = true;
		}

		if (IsSelectedMode && !inputEvent.HasShift)
		{
			IsSelectedMode = false;
		}

		switch (inputEvent.Key)
		{
			case ConsoleKey.LeftArrow:
				_editIndex = Math.Max(0, _editIndex - 1);
				break;

			case ConsoleKey.RightArrow:
				_editIndex = Math.Min(Value.Length, _editIndex + 1);
				break;

			case ConsoleKey.Backspace:
				_editIndex = Math.Max(0, _editIndex - 1);
				newText = $"{Value.Substring(0, _editIndex)}{Value.SubStr(_editIndex + 1)}";
				break;

			case ConsoleKey.Delete:
				newText = $"{Value.Substring(0, _editIndex)}{Value.SubStr(_editIndex + 1)}";
				break;

			case ConsoleKey.Home:
				_editIndex = 0;
				break;

			case ConsoleKey.End:
				_editIndex = Value.Length;
				break;

			default:
				var character = inputEvent.Key == ConsoleKey.Enter
					 ? '\n'
					 : inputEvent.KeyChar;
				newText = $"{Value.Substring(0, _editIndex)}{character}{Value.SubStr(_editIndex)}";
				_editIndex = Math.Min(newText.Length, _editIndex + 1);
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

public struct StrSpan
{
	public static StrSpan Empty = new StrSpan
	{
		Index = 0,
		Length = 0
	};

	public int Index { get; init; }
	public bool IsEmpty => (Index == 0 && Length == 0);
	public int Length { get; init; }

	public int Right => Index + Length - 1;
	public bool Contain(int pos)
	{
		return (pos >= Index && pos < Index + Length);
	}

	public StrSpan Intersect(StrSpan b)
	{
		if (this.IsEmpty || b.IsEmpty) return Empty;
		return new StrSpan
		{
			Index = Math.Max(this.Index, b.Index),
			Length = Math.Min(this.Right, b.Right) - Math.Max(this.Index, b.Index) + 1,
		};
	}
	public override string ToString()
	{
		return $"{nameof(StrSpan)}{{{Index},{Length}}}";
	}
}