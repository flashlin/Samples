namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleTextBox : IConsoleElement
{
	private int _caretStart;
	private int _caretEnd;
	private int _editIndex;

	public ConsoleTextBox(Rect rect)
	{
		GetViewRect = () => rect;
		EditRect = rect;
	}

	public string Value { get; set; } = String.Empty;
	public int MaxLength { get; set; } = int.MaxValue;
	public Rect EditRect { get; set; }
	public Func<Rect> GetViewRect { get; set; }
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
	public int EditIndex
	{
		get { return _editIndex; }
	}

	public Character this[Position pos]
	{
		get
		{
			var rect = EditRect.Intersect(GetViewRect());
			if (!rect.Contain(pos))
			{
				return Character.Empty;
			}

			var (startIndex, len) = ComputeShowContent(rect);
			var content = Value.Substring(startIndex, len);

			var x = pos.X - rect.Left;
			if (x >= content.Length)
			{
				return Character.Empty;
			}
			return new Character(content[x]);
		}
	}

	public bool OnInput(InputEvent inputEvent)
	{
		var rect = EditRect.Intersect(GetViewRect());
		var newText = (string?)null;
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
			//  break;
			// case ConsoleKey key when char.IsControl(inputEvent.Key.KeyChar) && inputEvent.Key.Key != ConsoleKey.Enter:
			//     return;
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

	private void MoveCursorRight(Rect rect, string? value)
	{
		var valueLength = 0;
		if (value != null)
		{
			valueLength = value.Length;
		}
		_caretEnd = Math.Min(_caretEnd + 1, valueLength);

		if (_caretEnd >= rect.Width)
		{
			_caretStart = Math.Max(_caretStart + 1, _caretEnd - rect.Width - 1);
		}
		_editIndex = Math.Max(Value.Length, _editIndex + 1);
	}

	private (int startIndex, int len) ComputeShowContent(Rect rect)
	{
		var startIndex = _editIndex - rect.Width;
		if (_editIndex < rect.Width)
		{
			startIndex = 0;
		}
		var len = Math.Min(Value.Length, rect.Width);
		return (startIndex, len);
	}
}

public static class StringExtension
{
	public static string SubStr(this string str, int offset)
	{
		if (string.IsNullOrEmpty(str))
		{
			return string.Empty;
		}
		if (offset < 0)
		{
			return string.Empty;
		}
		if (offset >= str.Length)
		{
			return string.Empty;
		}
		return str.Substring(offset);
	}
}