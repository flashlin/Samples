using System.Collections.ObjectModel;
using System.Collections.Specialized;

namespace GitCli.Models.ConsoleMixedReality;


public class HorizontalStack : IConsoleElement
{
	private int _focusIndex = -1;
	private IConsoleWriter _console;

	public HorizontalStack()
	{
		Children = new StackChildren(this);
	}

	public StackChildren Children { get; private set; }

	public bool FixedLayout { get; set; } = false;

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

	public void OnCreate(Rect rect, IConsoleWriter console)
	{
		_console = console;
		var noInitViewRect = ViewRect.IsEmpty;
		var viewRect = ViewRect = ViewRect.Init(() => rect);
		RearrangeChildren(viewRect, noInitViewRect);

		if (!FixedLayout && noInitViewRect)
		{
			RearrangeChildrenByChildWidth();
		}
	}

	private void RearrangeChildrenByChildWidth()
	{
		var prevRect = Rect.Empty;
		Children.ForEachIndex((child, idx) =>
		{
			if (idx == 0)
			{
				prevRect = child.ViewRect = child.GetChildrenRect();
				return;
			}

			var childRect = child.GetChildrenRect();
			child.ViewRect = new Rect
			{
				Left = prevRect.Right + 1,
				Top = childRect.Top,
				Width = childRect.Width,
				Height = childRect.Height
			};
		});
	}

	private void RearrangeChildren(Rect viewRect, bool noInitViewRect)
	{
		var left = viewRect.Left;
		var everyWidth = viewRect.Width / Children.Count;
		Children.ForEachIndex((child, idx) =>
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
				Width = noInitViewRect ? Math.Max(child.ViewRect.Width, everyWidth) : child.ViewRect.Width,
				Height = Math.Max(child.ViewRect.Height, viewRect.Height),
			};
			child.OnCreate(viewRect, _console);
			left += child.ViewRect.Width;
		});
	}

	public Rect GetChildrenRect()
	{
		return Children.GetRect();
	}

	public bool OnInput(InputEvent inputEvent)
	{
		return Children.GetFocusedControl().OnInput(inputEvent);
	}
}