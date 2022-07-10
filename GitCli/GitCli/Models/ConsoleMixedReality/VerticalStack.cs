namespace GitCli.Models.ConsoleMixedReality;

public class VerticalStack : IConsoleElement
{
	private int _focusIndex = -1;

	public VerticalStack()
	{
	}

	public List<IConsoleElement> Children { get; set; } = new List<IConsoleElement>();
	public Position CursorPosition
	{
		get
		{
			GetFocusedControl();

			var focus = GetFocusedControl();
			if (focus != null)
			{
				return focus.CursorPosition;
			}

			return ViewRect.BottomRightCorner;
		}
	}

	public bool IsTab { get; set; }
	public IConsoleElement? Parent { get; set; }
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
			JumpUpToChild();
			return;
		}

		if (inputEvent.Key == ConsoleKey.Tab)
		{
			JumpDownToChild();
			return;
		}

		if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
		{
			if (_focusIndex != -1)
			{
				JumpUpToChild();
				return;
			}

			Parent?.OnBubbleEvent(this, inputEvent);
			return;
		}

		if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
		{
			if (_focusIndex != -1)
			{
				JumpDownToChild();
				return;
			}
		}

		Parent?.OnBubbleEvent(this, inputEvent);
	}

	public void OnCreate(Rect rect, IConsoleWriter console)
	{
		var viewRect = ViewRect = ViewRect.Init(() => rect);
		var top = viewRect.Top;
		foreach (var (child, idx) in Children.Select((val, idx) => (val, idx)))
		{
			if (idx == 0)
			{
				_focusIndex = 0;
				top = viewRect.Top + child.ViewRect.Top;
			}
			child.Parent = this;
			child.ViewRect = new Rect
			{
				Left = viewRect.Left + child.ViewRect.Left,
				Top = top,
				Width = child.ViewRect.Width,
				Height = child.ViewRect.Height,
			};
			child.OnCreate(rect, console);
			top += child.ViewRect.Height;
		}
	}

	public bool OnInput(InputEvent inputEvent)
	{
		var focus = GetFocusedControl();
		if (focus == null)
		{
			return false;
		}

		var handle = focus.OnInput(inputEvent);
		return handle;
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

	private IConsoleElement? GetFocusedControl()
	{
		if (_focusIndex == -1)
		{
			return null;
		}
		return Children[_focusIndex];
	}
	private void JumpDownToChild()
	{
		_focusIndex = Math.Min(_focusIndex + 1, Children.Count - 1);
	}

	private void JumpUpToChild()
	{
		_focusIndex = Math.Min(_focusIndex - 1, 0);
	}
}