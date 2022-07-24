using System.Collections.ObjectModel;
using System.Collections.Specialized;

namespace GitCli.Models.ConsoleMixedReality;


public class HorizontalStack : IConsoleElement
{
	public HorizontalStack()
	{
		Children = new StackChildren(this);
	}

	public Color BackgroundColor { get; set; } = ConsoleColor.DarkBlue;
	public StackChildren Children { get; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public object? DataContext { get; set; }
	public Color? HighlightBackgroundColor { get; set; }

	public Position CursorPosition => Children.GetFocusedControl().CursorPosition;
	public Rect DesignRect { get; set; } = Rect.Empty;
	public bool FixedLayout { get; set; } = false;
	public bool IsTab { get; set; } = true;
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

	public bool OnBubbleKeyEvent(IConsoleElement element, InputEvent inputEvent)
	{
		if (inputEvent.Key == ConsoleKey.Tab && inputEvent.HasShift)
		{
			if (Children.JumpUpFocus())
			{
				ConsoleManager.FocusedElement = Children.GetFocusedControl();
				Refresh();
				return true;
			}
			return Parent?.OnBubbleKeyEvent(element, inputEvent) ?? false;
		}

		if (inputEvent.Key == ConsoleKey.Tab)
		{
			if (Children.JumpDownFocus())
			{
				ConsoleManager.FocusedElement = Children.GetFocusedControl();
				Refresh();
				return true;
			}
			return Parent?.OnBubbleKeyEvent(element, inputEvent) ?? false;
		}

		if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
		{
			if (Children.JumpUpFocus())
			{
				ConsoleManager.FocusedElement = Children.GetFocusedControl();
				Refresh();
				return true;
			}
			return Parent?.OnBubbleKeyEvent(element, inputEvent) ?? false;
		}

		if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
		{
			if (Children.JumpDownFocus())
			{
				ConsoleManager.FocusedElement = Children.GetFocusedControl();
				Refresh();
				return true;
			}
			return Parent?.OnBubbleKeyEvent(element, inputEvent) ?? false;
		}

		return Parent?.OnBubbleKeyEvent(element, inputEvent) ?? false;
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		this.HandleOnCreate(rect, consoleManager);
		UpdateChildren((viewRect, child) =>
		{
			child.OnCreate(viewRect, ConsoleManager);
		});
	}

	public string Value { get; set; }

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

	public bool OnBubbleEvent(IConsoleElement element, ConsoleElementEvent evt)
	{
		return Parent.RaiseOnBubbleEvent(this, evt);
	}

	private void UpdateChildren(Action<Rect, IConsoleElement> updateChild)
	{
		var left = ViewRect.Left;
		var everyWidth = this.GetDesignRectOrViewValue(x=> x.Width) / Children.Count;
		Children.ForEachIndex((child, idx) =>
		{
			if (idx == 0)
			{
				left = ViewRect.Left + child.DesignRect.Left;
			}
			child.Parent = this;
			var childViewRect = new Rect
			{
				Left = left,
				Top = ViewRect.Top,
				Width = everyWidth,
				Height = Math.Max(child.DesignRect.Height, ViewRect.Height),
			};
			updateChild(childViewRect, child);
			left += child.ViewRect.Width;
		});
	}
}