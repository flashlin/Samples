using System.Collections.ObjectModel;
using System.Collections.Specialized;

namespace GitCli.Models.ConsoleMixedReality;


public class HorizontalStack : IConsoleElement
{
	private int _focusIndex = -1;

	public HorizontalStack()
	{
		Children = new StackChildren(this);
	}

	public StackChildren Children { get; private set; }

	public Position CursorPosition => Children.GetFocusedControl().CursorPosition;

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

			return Children.GetContent(pos);
		}
	}

	public void OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
		if (inputEvent.Key == ConsoleKey.Tab && inputEvent.HasShift)
		{
			Children.JumpUpFocus();
			return;
		}

		if (inputEvent.Key == ConsoleKey.Tab)
		{
			Children.JumpDownFocus();
			return;
		}

		if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
		{
			if (_focusIndex != -1)
			{
				Children.JumpUpFocus();
				return;
			}

			Parent?.OnBubbleEvent(this, inputEvent);
			return;
		}

		if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
		{
			if (_focusIndex != -1)
			{
				Children.JumpDownFocus();
				return;
			}
		}

		Parent?.OnBubbleEvent(this, inputEvent);
	}

	public void OnCreate(Rect rect)
	{
		var viewRect = ViewRect = ViewRect.Init(() => rect);
		var left = ViewRect.Left;
		var everyWidth = rect.Width / Children.Count;
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
				Top = viewRect.Top + child.ViewRect.Top,
				Width = Math.Max(child.ViewRect.Width, everyWidth),
				Height = Math.Max(child.ViewRect.Height, rect.Height),
			};
			child.OnCreate(rect);
			left += child.ViewRect.Width;
		}
	}

	public bool OnInput(InputEvent inputEvent)
	{
		return Children.GetFocusedControl().OnInput(inputEvent);
	}
}