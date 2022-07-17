namespace GitCli.Models.ConsoleMixedReality;

public class VerticalStack : IConsoleElement
{
	public VerticalStack()
	{
		Children = new StackChildren();
	}

	public Color BackgroundColor { get; set; } = ConsoleColor.Cyan;
	public StackChildren Children { get; private set; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Position CursorPosition => Children.GetFocusedControl().CursorPosition;
	public Rect DesignRect { get; set; } = Rect.Empty;
	public bool IsTab { get; set; }
	public string Name { get; set; }
	public IConsoleElement? Parent { get; set; }
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

	public void OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
		if (inputEvent.Key == ConsoleKey.Tab && inputEvent.HasShift)
		{
			if (!Children.JumpUpFocus())
			{
				Parent?.OnBubbleEvent(this, inputEvent);
				return;
			}

			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			ConsoleManager.FocusedElement!.Refresh();
			return;
		}

		if (inputEvent.Key == ConsoleKey.Tab)
		{
			if (!Children.JumpDownFocus())
			{
				Parent?.OnBubbleEvent(this, inputEvent);
				return;
			}

			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			ConsoleManager.FocusedElement!.Refresh();
			return;
		}

		if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
		{
			if (!Children.JumpUpFocus())
			{
				Parent?.OnBubbleEvent(this, inputEvent);
				return;
			}

			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			return;
		}

		if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
		{
			if (!Children.JumpDownFocus())
			{
				Parent?.OnBubbleEvent(this, inputEvent);
				return;
			}

			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			return;
		}

		Parent?.OnBubbleEvent(this, inputEvent);
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		this.HandleOnCreate(rect, consoleManager);
		RearrangeChildren();
	}

	public bool OnInput(InputEvent inputEvent)
	{
		return Children.GetFocusedControl().OnInput(inputEvent);
	}

	public void OnUpdate()
	{

	}

	public void Refresh()
	{
		RearrangeChildren();
	}
	private void RearrangeChildren()
	{
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
			child.OnCreate(childRect, ConsoleManager);
			top += child.ViewRect.Height;
		});
	}
}