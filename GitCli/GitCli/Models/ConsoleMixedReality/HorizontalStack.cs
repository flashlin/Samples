namespace GitCli.Models.ConsoleMixedReality;

public class HorizontalStack : IConsoleElement
{
	private int _focusIndex = -1;

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

	public void OnCreate(IConsoleManager manager)
	{
		var viewRect = ViewRect.Init(() => Rect.OfSize(manager.Console.GetSize()));
		var left = 0;
		foreach (var (child, idx) in Children.Select((val, idx) => (val, idx)))
		{
			if (idx == 0)
			{
				_focusIndex = 0;
				left = viewRect.Left + child.ViewRect.Left;
			}
			child.Parent = this;
			child.ViewRect = new Rect
			{
				Left = left,
				Top = viewRect.Left + child.ViewRect.Left,
				Width = child.ViewRect.Width,
				Height = child.ViewRect.Height,
			};
			child.OnCreate(manager);
			left += child.ViewRect.Width;
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