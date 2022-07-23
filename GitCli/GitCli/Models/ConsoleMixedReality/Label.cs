using Microsoft.EntityFrameworkCore.Metadata.Internal;

namespace GitCli.Models.ConsoleMixedReality;

public class Label : IConsoleElement
{
	public Label()
	{
		Children = new StackChildren(this);
	}

	public Color Background { get; set; } = ConsoleColor.DarkBlue;
	public Color BackgroundColor { get; set; } = ConsoleColor.DarkBlue;
	public StackChildren Children { get; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public object? DataContext { get; set; }
	public Color? HighlightBackgroundColor { get; set; }
	public Position CursorPosition => Position.Empty;
	public Rect DesignRect { get; set; } = new()
	{
		Height = 1,
	};
	public bool Enabled { get; set; }
	public bool IsTab { get; set; }
	public string Name { get; set; } = string.Empty;
	public IConsoleElement? Parent { get; set; }
	public string Value { get; set; } = String.Empty;
	public Rect ViewRect { get; set; }
	public Character this[Position pos]
	{
		get
		{
			if (!ViewRect.Contain(pos))
			{
				return Character.Empty;
			}

			var x = pos.X - ViewRect.Left;
			var text = Value.SubStr(x, 1);
			if (string.IsNullOrEmpty(text))
			{
				return new Character(' ', null, Background);
			}
			return new Character(text[0], null, Background);
		}
	}

	public bool OnBubbleKeyEvent(IConsoleElement element, InputEvent inputEvent)
	{
		return false;
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		ViewRect = DesignRect.ToViewRect(rect, consoleManager);
		ViewRect = ViewRect.ExtendBy(DesignRect.TopLeftCorner);
	}

	public bool OnInput(InputEvent inputEvent)
	{
		return false;
	}

	public void Refresh()
	{
	}

	public bool OnBubbleEvent(IConsoleElement element, ConsoleElementEvent evt)
	{
		return Parent.RaiseOnBubbleEvent(this, evt);
	}
}