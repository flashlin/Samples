using System;

namespace T1.ConsoleUiMixedReality;

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
	public Position CursorPosition => Position.Empty;
	public object? DataContext { get; set; }
	public Rect DesignRect { get; set; } = new()
	{
		Height = 1,
	};

	public bool Enabled { get; set; }
	public Color? HighlightBackgroundColor { get; set; }
	public bool IsTab { get; set; }
	public string Name { get; set; } = string.Empty;
	public IConsoleElement? Parent { get; set; }
	public object? UserObject { get; set; }
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

	public bool OnBubbleEvent(IConsoleElement element, ConsoleElementEvent evt)
	{
		return Parent.RaiseOnBubbleEvent(this, evt);
	}

	public bool OnBubbleKeyEvent(IConsoleElement element, InputEvent inputEvent)
	{
		return false;
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		this.HandleOnCreate(rect, consoleManager);
		Refresh();
	}

	public bool OnInput(InputEvent inputEvent)
	{
		return false;
	}

	public void Refresh()
	{
		if (DesignRect.IsEmpty)
		{
			var width = Value.Length;
			ViewRect = new Rect()
			{
				Left = ViewRect.Left,
				Top = ViewRect.Top,
				Width = width,
				Height = ViewRect.Height,
			};
		}
	}
}