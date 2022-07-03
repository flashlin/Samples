namespace GitCli.Models.ConsoleMixedReality;

public class VerticalStack : ConsoleControl
{
	private IConsoleElement? _focus;

	public override Position CursorPosition
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

	public override Character this[Position pos]
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

	public override void OnCreated()
	{
		var viewRect = ViewRect.Init(() => Rect.OfSize(VConsole!.GetSize()));
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

	public override bool OnInput(InputEvent inputEvent)
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