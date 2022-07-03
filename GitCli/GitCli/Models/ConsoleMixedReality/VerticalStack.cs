namespace GitCli.Models.ConsoleMixedReality;

public class VerticalStack : IConsoleElement
{
	private IConsoleElement? _focus;
	private readonly IConsoleWriter _console;

	public VerticalStack(IConsoleWriter console)
	{
		_console = console;
	}
	
	public Rect ViewRect { get; set; } = Rect.Empty;
	
	public List<IConsoleElement> Children { get; set; } = new List<IConsoleElement>();

	public virtual Position CursorPosition
	{
		get
		{
			_focus ??= Children.FirstOrDefault();

			if (_focus != null)
			{
				return _focus.CursorPosition;
			}

			return ViewRect.BottomRightCorner;
		}
	}

	public Character this[Position pos]
	{
		get
		{
			if (!ViewRect.Contain(pos))
			{
				return Character.Empty;
			}

			var character = new Character(' ', null, ConsoleColor.DarkGray);
			foreach (var child in Children)
			{
				var ch = child[pos];
				if (!ch.IsEmpty)
				{
					character = ch;
				}
			}
			return character;
		}
	}

	public void OnCreated()
	{
		var viewRect = ViewRect.Init(() => Rect.OfSize(_console.GetSize()));
		var top = 0;
		foreach (var (child, idx) in Children.Select((val, idx) => (val, idx)))
		{
			if (idx == 0)
			{
				top = viewRect.Top + child.ViewRect.Top;
			}
			child.ViewRect = new Rect
			{
				Left = viewRect.Left + child.ViewRect.Left,
				Top = viewRect.Top + child.ViewRect.Top,
				Width = child.ViewRect.Width,
				Height = child.ViewRect.Height,
			};
			top += child.ViewRect.Height;
		}
	}

	public bool OnInput(InputEvent inputEvent)
	{
		var child = Children
			 .FirstOrDefault(x => x.OnInput(inputEvent));
		if (child != null)
		{
			return true;
		}

		return false;
	}
}