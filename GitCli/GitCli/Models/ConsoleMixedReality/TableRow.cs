using System.Collections.ObjectModel;

namespace GitCli.Models.ConsoleMixedReality;

public class TableRow : IConsoleElement
{
	public TableRow(Rect rect, StackChildren children)
	{
		DesignRect = rect;
		Children = children;
	}

	public Color BackgroundColor { get; set; } = ConsoleColor.DarkBlue;

	public StackChildren Children { get; private set; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Position CursorPosition => Children.GetFocusedControl().CursorPosition;
	public object? DataContext { get; set; }
	public Rect DesignRect { get; set; }
	public Color? HighlightBackgroundColor { get; set; }
	public bool IsTab { get; set; }

	public string Name { get; set; } = string.Empty;

	public IConsoleElement? Parent { get; set; }

	public object? UserObject { get; set; }
	public string Value { get; set; }
	public Rect ViewRect { get; set; }

	public Character this[Position pos]
	{
		get
		{
			foreach (var child in Children)
			{
				var ch = child[pos];
				if (!ch.IsEmpty)
				{
					return ch;
				}
			}
			return Character.Empty;
		}
	}

	public bool OnBubbleEvent(IConsoleElement element, ConsoleElementEvent evt)
	{
		return Parent.RaiseOnBubbleEvent(this, evt);
	}

	public bool OnBubbleKeyEvent(IConsoleElement element, InputEvent inputEvent)
	{
		return false;
	}

	public void OnCreate(Rect parentRect, IConsoleManager consoleManager)
	{
		ViewRect = DesignRect.ToViewRect(parentRect, consoleManager);
		var columnWidth = ViewRect.Width / Children.Count;
		var left = ViewRect.Left;
		foreach (var child in Children.WithIndex())
		{
			var childRect = new Rect
			{
				Left = left,
				Top = ViewRect.Top,
				Width = columnWidth,
				Height = ViewRect.Height,
			};
			child.val.DesignRect = new Rect()
			{
				Left = 0,
				Top = 0,
				Width = columnWidth,
				Height = 1,
			};
			child.val.OnCreate(childRect, consoleManager);
			left += columnWidth;
		}
	}
	public bool OnInput(InputEvent inputEvent)
	{
		return Children.GetFocusedControl().OnInput(inputEvent);
	}

	public void Refresh()
	{
	}
}