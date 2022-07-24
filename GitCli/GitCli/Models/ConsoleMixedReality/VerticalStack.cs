namespace GitCli.Models.ConsoleMixedReality;

public class VerticalStack : IConsoleElement
{
	public VerticalStack()
	{
		Children = new StackChildren(this);
	}

	public Color BackgroundColor { get; set; } = ConsoleColor.Cyan;
	public StackChildren Children { get; private set; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Position CursorPosition => Children.GetFocusedControl().CursorPosition;
	public object? DataContext { get; set; }
	public Rect DesignRect { get; set; } = Rect.Empty;
	public Color? HighlightBackgroundColor { get; set; }
	public bool IsTab { get; set; } = true;
	public string Name { get; set; } = string.Empty;
	public IConsoleElement? Parent { get; set; }
	public string Value { get; set; }
	public object? UserObject { get; set; }
	public Rect ViewRect { get; set; } = Rect.Empty;
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

	public Rect GetChildrenRect()
	{
		var initRect = Rect.Empty;
		foreach (var child in Children)
		{
			initRect = initRect.Surround(child.ViewRect);
		}
		return initRect;
	}

	public bool OnBubbleEvent(IConsoleElement element, ConsoleElementEvent evt)
	{
		return Parent.RaiseOnBubbleEvent(this, evt);
	}

	public bool OnBubbleKeyEvent(IConsoleElement element, InputEvent inputEvent)
	{
		if (inputEvent.Key == ConsoleKey.Tab && inputEvent.HasShift)
		{
			if (!Children.JumpUpFocus())
			{
				return Parent?.OnBubbleKeyEvent(this, inputEvent) ?? false;
			}

			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			Refresh();
			return true;
		}

		if (inputEvent.Key == ConsoleKey.Tab)
		{
			if (!Children.JumpDownFocus())
			{
				return Parent?.OnBubbleKeyEvent(this, inputEvent) ?? false;
			}

			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			Refresh();
			return true;
		}

		if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
		{
			if (!Children.JumpUpFocus())
			{
				return Parent?.OnBubbleKeyEvent(this, inputEvent) ?? false;
			}

			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			return true;
		}

		if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
		{
			if (!Children.JumpDownFocus())
			{
				Parent?.OnBubbleKeyEvent(this, inputEvent);
				return false;
			}

			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			return true;
		}

		return Parent?.OnBubbleKeyEvent(this, inputEvent) ?? false;
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		this.HandleOnCreate(rect, consoleManager);
		UpdateChildren((viewRect, child) =>
		{
			child.OnCreate(viewRect, ConsoleManager);
		});
	}
	public bool OnInput(InputEvent inputEvent)
	{
		return Children.GetFocusedControl().OnInput(inputEvent);
	}

	public void Refresh()
	{
		UpdateChildren((viewRect, child) =>
		{
			child.ViewRect = viewRect;
			child.Refresh();
		});
	}
	private void UpdateChildren(Action<Rect, IConsoleElement> updateChild)
	{
		var top = ViewRect.Top;
		var everyWidth = this.GetDesignRectOrViewValue(x => x.Width);
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
				Width = everyWidth, //child.DesignRect.Width,
				Height = child.DesignRect.Height,
			};
			updateChild(childRect, child);
			top += child.ViewRect.Height;
		});
	}
}