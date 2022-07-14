namespace GitCli.Models.ConsoleMixedReality;

public class VerticalStack : IConsoleElement
{
	private IConsoleManager _consoleManager;

	public VerticalStack()
	{
		Children = new StackChildren();
	}

	public StackChildren Children { get; private set; }

	public Position CursorPosition => Children.GetFocusedControl().CursorPosition;

	public bool IsTab { get; set; }
	public IConsoleElement? Parent { get; set; }
	public Rect DesignRect { get; set; } = Rect.Empty;
	public Rect ViewRect { get; set; } = Rect.Empty;
	public Color BackgroundColor { get; set; } = ConsoleColor.Cyan;
	public Character this[Position pos]
	{
		get
		{
			if (!ViewRect.Contain(pos))
			{
				return Character.Empty;
			}

			var character = new Character(' ', null, BackgroundColor);
			foreach (var child in Children)
			{
				var ch = child[pos];
				if (!ch.IsEmpty)
				{
					character = ch;
					return ch;
				}
			}

			return character;
		}
	}

	public void OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
		if (inputEvent.Key == ConsoleKey.Tab && inputEvent.HasShift)
		{
			if (!Children.JumpUpFocus())
			{
				Parent?.OnBubbleEvent(this, inputEvent);
				return;
			}

			_consoleManager.FocusedElement = Children.GetFocusedControl();
			_consoleManager.FocusedElement!.Refresh();
			return;
		}

		if (inputEvent.Key == ConsoleKey.Tab)
		{
			if (!Children.JumpDownFocus())
			{
				Parent?.OnBubbleEvent(this, inputEvent);
				return;
			}

			_consoleManager.FocusedElement = Children.GetFocusedControl();
			_consoleManager.FocusedElement!.Refresh();
			return;
		}

		if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
		{
			if (!Children.JumpUpFocus())
			{
				Parent?.OnBubbleEvent(this, inputEvent);
				return;
			}

			_consoleManager.FocusedElement = Children.GetFocusedControl();
			return;
		}

		if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
		{
			if (!Children.JumpDownFocus())
			{
				Parent?.OnBubbleEvent(this, inputEvent);
				return;
			}

			_consoleManager.FocusedElement = Children.GetFocusedControl();
			return;
		}

		Parent?.OnBubbleEvent(this, inputEvent);
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		_consoleManager = consoleManager;
		ViewRect = DesignRect.ToViewRect(rect, _consoleManager);
		var top = ViewRect.Top;
		Children.ForEachIndex((child, idx) =>
		{
			if (idx == 0)
			{
				top = ViewRect.Top + child.DesignRect.Top;
			}
			child.Parent = this;
			var childRect = new Rect
			{
				Left = ViewRect.Left + child.DesignRect.Left,
				Top = top,
				Width = child.DesignRect.Width,
				Height = child.DesignRect.Height,
			};
			child.OnCreate(childRect, _consoleManager);
			top += child.ViewRect.Height;
		});
	}

	public bool OnInput(InputEvent inputEvent)
	{
		return Children.GetFocusedControl().OnInput(inputEvent);
	}

	public Rect GetChildrenRect()
	{
		var initRect = Rect.Empty;
		foreach (var child in Children)
		{
			initRect = initRect.Surround(child.ViewRect);
		}
		return initRect;
	}

	public void Refresh()
	{
		//var prevRect = Rect.Empty;
		//Children.ForEachIndex((child, idx) =>
		//{
		//	if (idx == 0)
		//	{
		//		prevRect = child.ViewRect = child.GetChildrenRect();
		//		return;
		//	}

		//	var childRect = child.GetChildrenRect();
		//	childRect = new Rect
		//	{
		//		Left = childRect.Left,
		//		Top = prevRect.Bottom + 1,
		//		Width = childRect.Width,
		//		Height = childRect.Height
		//	};
		//	child.OnCreate(childRect, _consoleManager);
		//});
	}
}