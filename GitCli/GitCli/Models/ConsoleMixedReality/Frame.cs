namespace GitCli.Models.ConsoleMixedReality;

public class Frame : IConsoleElement
{
	private IConsoleManager _consoleManager;

	private IConsoleElement? _focus;

	public Frame(Rect viewRect)
	{
        ViewRect = viewRect;
   }

	public Color BackgroundColor { get; set; } = ConsoleColor.DarkBlue;
	public StackChildren Children { get; set; } = new();
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Position CursorPosition => Children.GetFocusedControl().CursorPosition;

	public Rect DesignRect { get; set; } = Rect.Empty;
	public bool IsTab { get; set; } = false;
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

	public Rect GetChildrenRect()
	{
		var initRect = Rect.Empty;
		foreach (var child in this.Children)
		{
			initRect = initRect.Surround(child.ViewRect);
		}
		return initRect;
	}

	public void OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
		var focus = Children.GetFocusedControl();
		
		if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
		{
			Children.JumpUpFocus();
			return;
		}

		if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
		{
			Children.JumpDownFocus();
			return;
		}

		Parent?.OnBubbleEvent(element, inputEvent);
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		_consoleManager = consoleManager;
		var viewRect = ViewRect.Init(() => rect);
		foreach (var (child, idx) in Children.Select((val, idx) => (val, idx)))
		{
			if (idx == 0)
			{
				_focus = child;
			}

			child.Parent = this;
			child.ViewRect = new Rect
			{
				Left = viewRect.Left + child.ViewRect.Left,
				Top = viewRect.Top + child.ViewRect.Top,
				Width = child.ViewRect.Width,
				Height = child.ViewRect.Height,
			};
			child.OnCreate(rect, consoleManager);
		}
	}

	public bool OnInput(InputEvent inputEvent)
	{
		if (_focus == null)
		{
			return false;
		}

		return _focus.OnInput(inputEvent);
	}

	public void OnUpdate()
	{
	}

	public void Refresh()
	{
	}
}