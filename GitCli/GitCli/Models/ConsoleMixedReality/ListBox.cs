using System.Collections;
using System.Collections.ObjectModel;
using System.Collections.Specialized;

namespace GitCli.Models.ConsoleMixedReality;

[MapClone]
public class ListBox : IConsoleElement
{
	private int _index = -1;
	private Span _showListItemSpan = Span.Empty;

	public ListBox(Rect rect)
	{
		DesignRect = rect;
		Children.CollectionChanged += ChildrenOnCollectionChanged;
	}

	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Color BackgroundColor { get; set; } = ConsoleColor.Blue;
	public StackChildren Children { get; } = new();
	public Position CursorPosition => Children.GetFocusedControl().CursorPosition;

	public Rect DesignRect { get; set; }
	public bool IsTab { get; set; } = true;
	public int MaxLength { get; set; } = int.MaxValue;
	public string Name { get; set; }
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

			if (_showListItemSpan.IsEmpty)
			{
				return new Character(' ', null, BackgroundColor);
			}

			var y = pos.Y - ViewRect.Top;
			var index = _showListItemSpan.Index + y;

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
		_index = Math.Max(_index, 0);
		var textBox = new TextBox(Rect.Empty)
		{
			Value = item.Title,
			UserObject = item.Value
		};
		Children.Add(textBox);
		return textBox;
	}

	public Rect GetChildrenRect()
	{
		return ViewRect;
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

		OnUpdate();
		_showListItemSpan = new Span()
		{
			Index = 0,
			Length = ViewRect.Height
		};
	}

	public bool OnInput(InputEvent inputEvent)
	{
		var focusedControl = Children.GetFocusedControl();

		switch (inputEvent.Key)
		{
			case ConsoleKey.Tab:
				focusedControl.OnInput(inputEvent);
				OnUpdate();
				return true;

			case ConsoleKey.LeftArrow:
				focusedControl.OnInput(inputEvent);
				OnUpdate();
				return true;

			case ConsoleKey.RightArrow:
				focusedControl.OnInput(inputEvent);
				OnUpdate();
				return true;

			case ConsoleKey.UpArrow when !inputEvent.HasControl:
				Children.JumpUpFocus();
				ConsoleManager.FocusedElement = Children.GetFocusedControl();
				OnUpdate();
				break;

			case ConsoleKey.DownArrow when !inputEvent.HasControl:
				Children.JumpDownFocus();
				ConsoleManager.FocusedElement = Children.GetFocusedControl();
				OnUpdate();
				break;

			case ConsoleKey.Enter:
				focusedControl.OnInput(inputEvent);
				break;
		}

		return true;
	}

	public void OnUpdate()
	{
		var y = ViewRect.Top;
		foreach (var child in Children)
		{
			_index = 0;
			child.ViewRect = new Rect()
			{
				Left = ViewRect.Left,
				Top = y,
				Width = ViewRect.Width,
				Height = 1,
			};
			child.BackgroundColor = GetHighlightBackgroundColor(child);
			child.OnUpdate();
			y += 1;
		}
	}

	public void Refresh()
	{
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
		return ConsoleManager.FocusedElement == child ?
			ConsoleManager.HighlightBackgroundColor1 :
			ConsoleManager.HighlightBackgroundColor2;
	}
}

