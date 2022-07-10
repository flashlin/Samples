namespace GitCli.Models.ConsoleMixedReality;

public class VerticalStack : IConsoleElement
{
	public VerticalStack()
	{
		Children = new StackChildren();
	}

	public StackChildren Children { get; private set; }

	public Position CursorPosition
	{
		get
		{
			return Children.GetFocusedControl().CursorPosition;
			//var focus = GetFocusedControl();
			//if (focus != null)
			//{
			//	return focus.CursorPosition;
			//}
			//return ViewRect.BottomRightCorner;
		}
	}

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
			return;
		}

		if (inputEvent.Key == ConsoleKey.Tab)
		{
			if (!Children.JumpDownFocus())
			{
				Parent?.OnBubbleEvent(this, inputEvent);
				return;
			}
			return;
		}

		if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
		{
			if (!Children.JumpUpFocus())
			{
				Parent?.OnBubbleEvent(this, inputEvent);
				return;
			}
			return;
		}

		if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
		{
			if (!Children.JumpDownFocus())
			{
				Parent?.OnBubbleEvent(this, inputEvent);
				return;
			}
			return;
		}

		Parent?.OnBubbleEvent(this, inputEvent);
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		ViewRect = DesignRect.ToViewRect(rect);

		var top = ViewRect.Top;
		Children.ForEachIndex((child, idx) =>
		{
			if (idx == 0)
			{
				//_focusIndex = 0;
				top = ViewRect.Top + child.ViewRect.Top;
			}

			child.Parent = this;

			var childRect = new Rect
			{
				Left = ViewRect.Left + child.ViewRect.Left,
				Top = top,
				Width = child.ViewRect.Width,
				Height = child.ViewRect.Height,
			};
			child.OnCreate(childRect, consoleManager);
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
}