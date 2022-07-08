namespace GitCli.Models.ConsoleMixedReality;

public class VerticalStack : IConsoleElement
{
	private int _focusIndex = -1;

	public Rect ViewRect { get; set; } = Rect.Empty;

	public IConsoleElement? Parent { get; set; }
	public bool IsTab { get; set; }

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

	private IConsoleElement? GetFocusedControl()
	{
		if (_focusIndex == -1)
		{
			return null;
		}
		return Children[_focusIndex];
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

	public void OnCreate(IConsoleManager manager)
	{
		var viewRect = ViewRect.Init(() => Rect.OfSize(manager.Console.GetSize()));
		var top = 0;
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
			child.OnCreate(manager);
			top += child.ViewRect.Height;
		}
	}

	public void OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
		if (inputEvent.Key == ConsoleKey.Tab)
		{
			DownJumpToChild();
			return;
		}

		var focus = GetFocusedControl();
		if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
		{
			if (_focusIndex != -1)
			{
				_focusIndex = Math.Min(_focusIndex - 1, 0);
				return;
			}
			//if (focus != null)
			//{
			//    var idx = Children.FindIndex(x => x == focus);
			//    idx = Math.Min(idx - 1, 0);
			//    _focusIndex = idx;
			//    return;
			//}

			Parent?.OnBubbleEvent(this, inputEvent);
			return;
		}

		if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
		{
			if (_focusIndex != -1)
			{
				DownJumpToChild();
				return;
			}
			//if (focus != null)
			//{
			//	var idx = Children.FindIndex(x => x == focus);
			//	idx = Math.Min(idx + 1, Children.Count - 1);
			//	_focusIndex = idx;
			//	return;
			//}
		}

		Parent?.OnBubbleEvent(this, inputEvent);
	}

	private void DownJumpToChild()
	{
		_focusIndex = Math.Min(_focusIndex + 1, Children.Count - 1);
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
}