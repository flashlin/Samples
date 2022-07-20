using System.Collections.ObjectModel;
using System.Collections.Specialized;

namespace GitCli.Models.ConsoleMixedReality;


public class HorizontalStack : IConsoleElement
{
	public HorizontalStack()
	{
		Children = new StackChildren();
	}

	public Color BackgroundColor { get; set; } = ConsoleColor.DarkBlue;
	public StackChildren Children { get; private set; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;

	public Position CursorPosition => Children.GetFocusedControl().CursorPosition;
	public Rect DesignRect { get; set; } = Rect.Empty;
	public bool FixedLayout { get; set; } = false;
	public bool IsTab { get; set; }
	public string Name { get; set; } = String.Empty;
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

	public Rect GetChildrenRect()
	{
		return Children.GetRect();
	}

	public void OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
		if (inputEvent.Key == ConsoleKey.Tab && inputEvent.HasShift)
		{
			Children.JumpUpFocus();
			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			return;
		}

		if (inputEvent.Key == ConsoleKey.Tab)
		{
			Children.JumpDownFocus();
			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			return;
		}

		if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
		{
			Children.JumpUpFocus();
			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			//Parent?.OnBubbleEvent(this, inputEvent);
			return;
		}

		if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
		{
			Children.JumpDownFocus();
			ConsoleManager.FocusedElement = Children.GetFocusedControl();
			return;
		}

		Parent?.OnBubbleEvent(this, inputEvent);
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		this.HandleOnCreate(rect, consoleManager);

		var userInitDesignRect = DesignRect.IsEmpty;
		RearrangeChildren(ViewRect, userInitDesignRect);

		if (!FixedLayout && userInitDesignRect)
		{
			RearrangeChildrenByChildWidth();
		}
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
		RearrangeChildren(ViewRect, DesignRect.IsEmpty);
	}

	private void RearrangeChildren(Rect viewRect, bool userInitViewRect)
	{
		var left = viewRect.Left;
		var everyWidth = viewRect.Width / Children.Count;
		Children.ForEachIndex((child, idx) =>
		{
			if (idx == 0)
			{
				left = viewRect.Left + child.ViewRect.Left;
			}
			child.Parent = this;
			var childViewRect = new Rect
			{
				Left = left,
				Top = viewRect.Top,
				Width = userInitViewRect ? Math.Max(child.DesignRect.Width, everyWidth) : child.ViewRect.Width,
				Height = Math.Max(child.DesignRect.Height, viewRect.Height),
			};
			child.OnCreate(childViewRect, ConsoleManager);
			left += child.ViewRect.Width;
		});
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

			//child.ViewRect = childRect = new Rect
			child.ViewRect = Rect.Empty;
			childRect = new Rect
			{
				Left = prevRect.Right + 1,
				Top = childRect.Top,
				Width = childRect.Width,
				Height = childRect.Height
			};

			child.OnCreate(childRect, ConsoleManager);
		});
	}
}