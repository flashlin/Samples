using Microsoft.EntityFrameworkCore.Metadata.Internal;

namespace GitCli.Models.ConsoleMixedReality;

public class Label : IConsoleElement
{
	public Label(Rect rect)
	{
		DesignRect = rect;
	}

	public Color Background { get; set; } = ConsoleColor.DarkBlue;
	public Color BackgroundColor { get; set; } = ConsoleColor.DarkBlue;
	public StackChildren Children { get; } = new();
	public Position CursorPosition => Position.Empty;
	public Rect DesignRect { get; set; }
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

	public Rect GetChildrenRect()
	{
		return ViewRect;
	}

	public void OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
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

	public void OnUpdate()
	{
	}

	public void Refresh()
	{
	}
}