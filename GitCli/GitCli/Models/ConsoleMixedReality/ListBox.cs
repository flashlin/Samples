using System.Collections;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using LanguageExt;
using Microsoft.EntityFrameworkCore.Metadata.Internal;

namespace GitCli.Models.ConsoleMixedReality;

[MapClone]
public class ListBox : IConsoleElement
{
	private Span _showListSpan = Span.Empty;

	public ListBox(Rect rect)
	{
		DesignRect = rect;
		Children = new StackChildren(this);
		Children.CollectionChanged += ChildrenOnCollectionChanged;
	}

	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Color? HighlightBackgroundColor { get; set; }
	public Color BackgroundColor { get; set; } = ConsoleColor.Blue;
	public StackChildren Children { get; }
	public Position CursorPosition => Children.FocusedControlOrMe(
		x => x.CursorPosition,
		() => ViewRect.TopLeftCorner);

	public Rect DesignRect { get; set; }
	public bool IsTab { get; set; } = true;
	public int MaxLength { get; set; } = int.MaxValue;
	public string Name { get; set; } = string.Empty;
	public IConsoleElement? Parent { get; set; }

	public Rect ViewRect { get; set; }

	public Character this[Position pos]
	{
		get
		{
			if (!ViewRect.Contain(pos))
			{
				return Character.Empty;
			}

			if (_showListSpan.IsEmpty)
			{
				return new Character(' ', null, BackgroundColor);
			}

			var y = pos.Y - ViewRect.Top;
			var index = _showListSpan.Index + y;

			if (index >= Children.Count)
			{
				return new Character(' ', null, BackgroundColor);
			}

			var item = Children[index];
			return item[pos];
		}
	}

	public TextBox AddItem(ListItem item)
	{
		var textBox = new TextBox(Rect.Empty)
		{
			Parent = this,
			Value = item.Title,
			UserObject = item.Value
		};
		Children.Add(textBox);
		if (ConsoleManager.FocusedElement == this)
		{
			ConsoleManager.FocusedElement = textBox;
		}
		return textBox;
	}

	public void OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
		switch (inputEvent.Key)
		{
			case ConsoleKey.Tab:
				Parent?.OnBubbleEvent(element, inputEvent);
				break;
		}
	}

	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		this.HandleOnCreate(rect, consoleManager);
		Refresh();
		_showListSpan = new Span()
		{
			Index = 0,
			Length = ViewRect.Height
		};
	}

	public bool OnInput(InputEvent inputEvent)
	{
		return Children.FocusedControlOrMe(
			focusedControl => OnFocusedControlInputEvent(inputEvent, focusedControl),
			() => OnMeInputEvent(inputEvent));
	}

	private bool OnMeInputEvent(InputEvent inputEvent)
	{
		OnBubbleEvent(this, inputEvent);
		return true;
	}

	private bool OnFocusedControlInputEvent(InputEvent inputEvent, IConsoleElement focusedControl)
	{
		switch (inputEvent.Key)
		{
			case ConsoleKey.Tab:
				focusedControl.OnInput(inputEvent);
				Refresh();
				return true;

			case ConsoleKey.LeftArrow:
				focusedControl.OnInput(inputEvent);
				Refresh();
				return true;

			case ConsoleKey.Home:
			case ConsoleKey.End:
			case ConsoleKey.RightArrow:
				focusedControl.OnInput(inputEvent);
				Refresh();
				return true;

			case ConsoleKey.UpArrow when !inputEvent.HasControl:
				if (Children.JumpUpFocus() && 0 == CursorPosition.Y - ViewRect.Top)
				{
					_showListSpan = _showListSpan.Move(-1);
				}

				ConsoleManager.FocusedElement = Children.GetFocusedControl();
				Refresh();
				break;

			case ConsoleKey.DownArrow when !inputEvent.HasControl:
				if (Children.JumpDownFocus() && ViewRect.Height == CursorPosition.Y - ViewRect.Top)
				{
					_showListSpan = _showListSpan.Move(1);
				}

				ConsoleManager.FocusedElement = Children.GetFocusedControl();
				Refresh();
				break;

			case ConsoleKey.Enter:
				focusedControl.OnInput(inputEvent);
				break;
		}

		return true;
	}

	public void Refresh()
	{
		var y = ViewRect.Top;
		Children.ForEachIndex((child, idx) =>
		{
			if (idx < _showListSpan.Index)
			{
				return;
			}
			child.Parent = this;
			child.ViewRect = new Rect()
			{
				Left = ViewRect.Left,
				Top = y,
				Width = ViewRect.Width,
				Height = 1,
			};
			child.BackgroundColor = GetHighlightBackgroundColor(child);
			child.Refresh();
			y += 1;
		});
	}

	private void AddChild(IList newItems)
	{
		foreach (IConsoleElement item in newItems)
		{
			item.Parent = this;
		}
	}

	private void ChildrenOnCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
	{
		if (e.Action == NotifyCollectionChangedAction.Add)
		{
			AddChild(e.NewItems!);
		}
	}

	private Color GetHighlightBackgroundColor(IConsoleElement child)
	{
		if (ConsoleManager.FocusedElement == child)
		{
			return ConsoleManager.HighlightBackgroundColor1;
		}

		return Children.GetFocusedControl() == child ? ConsoleManager.HighlightBackgroundColor2 : BackgroundColor;
	}
}

