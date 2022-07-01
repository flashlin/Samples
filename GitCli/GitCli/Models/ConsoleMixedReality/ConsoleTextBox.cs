namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleTextBox : IConsoleElement
{
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
	public Color Background { get; set; } = ConsoleColor.Blue;
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
				return new Character(' ', Background, Background);
			}
			return new Character(content[x], null, Background);
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