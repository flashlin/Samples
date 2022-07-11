using System.Collections;
using System.Collections.ObjectModel;
using System.Collections.Specialized;

namespace GitCli.Models.ConsoleMixedReality;

public class ListBox : IConsoleElement
{
	private IConsoleManager _consoleManager;
	private int _index = -1;
	private int _maxLength;
	private Span _showListItemSpan = Span.Empty;

	public ListBox(Rect rect)
	{
		DesignRect = rect;
		Children.CollectionChanged += ChildrenOnCollectionChanged;
	}

	public Color BackgroundColor { get; set; } = ConsoleColor.Blue;
	public ObservableCollection<TextBox> Children { get; } = new();
	public Position CursorPosition
	{
		get
		{
			if (_index >= 0)
			{
				return Children[_index].CursorPosition;
			}

			return new Position(ViewRect.Left, ViewRect.Top);
		}
	}

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
			item.ViewRect = new Rect
			{
				Left = ViewRect.Left,
				Top = ViewRect.Top + y,
				Width = ViewRect.Width,
				Height = ViewRect.Height
			};
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
			case ConsoleKey.Enter:
				Parent?.OnBubbleEvent(element, inputEvent);
				break;
		}
	}
	public void OnCreate(Rect rect, IConsoleManager consoleManager)
	{
		ViewRect = new Rect()
		{
			Left = rect.Left + DesignRect.Left,
			Top = rect.Top + DesignRect.Top,
			Width = rect.IsEmpty ? DesignRect.Width : rect.Width,
			Height = rect.IsEmpty ? DesignRect.Height : rect.Height,
		};

		_consoleManager = consoleManager;

		var y = ViewRect.Top;
		foreach (var child in Children)
		{
			_index = 0;
			var childRect = new Rect()
			{
				Left = ViewRect.Left,
				Top = y,
				Width = ViewRect.Width,
				Height = 1,
			};
			child.OnCreate(childRect, _consoleManager);
			_maxLength = Math.Max(_maxLength, child.Value.Length);
			y += 1;
		}
		_showListItemSpan = new Span()
		{
			Index = 0,
			Length = ViewRect.Height
		};
		RearrangeChildrenIndex();

		_consoleManager.FocusedElement ??= this;
	}

	public bool OnInput(InputEvent inputEvent)
	{
		switch (inputEvent.Key)
		{
			case ConsoleKey.Tab:
				GetFocusedListItem().OnInput(inputEvent);
				return true;

			case ConsoleKey.LeftArrow:
				GetFocusedListItem().OnInput(inputEvent);
				RearrangeChildrenIndex();
				return true;

			case ConsoleKey.RightArrow:
				GetFocusedListItem().OnInput(inputEvent);
				RearrangeChildrenIndex();
				return true;

			case ConsoleKey.UpArrow when !inputEvent.HasControl:
				JumpUpToItem();
				break;

			case ConsoleKey.DownArrow when !inputEvent.HasControl:
				JumpDownToItem();
				break;

			// case ConsoleKey.Home:
			//     _editIndex = 0;
			//     break;
			//
			// case ConsoleKey.End:
			//     _editIndex = Value.Length;
			//     break;

			case ConsoleKey.Enter:
				GetFocusedListItem().OnInput(inputEvent);
				break;
		}

		return true;
	}

	private void AddChild(IList newItems)
	{
		foreach (IConsoleElement item in newItems)
		{
			item.Parent = this;
		}
	}

	private void AfterMove((int prevEditIndex, bool isEditEnd) info)
	{
		var focusedItem = GetFocusedListItem();
		if (info.isEditEnd)
		{
			focusedItem.EditIndex = focusedItem.Value.Length;
		}
		else
		{
			focusedItem.EditIndex = info.prevEditIndex;
		}
		RearrangeChildrenIndex();
	}

	private (int prevEditIndex, bool isEditEnd) BeforeMoveDown()
	{
		var focusedItem = GetFocusedListItem();
		var prevEditIndex = focusedItem.EditIndex;
		if (_index == _showListItemSpan.Right && _showListItemSpan.Right + 1 < Children.Count)
		{
			_showListItemSpan = _showListItemSpan.Move(1);
		}

		var isEditEnd = (prevEditIndex == focusedItem.Value.Length);
		return (prevEditIndex, isEditEnd);
	}

	private (int prevEditIndex, bool isEditEnd) BeforeMoveUp()
	{
		var focusedItem = GetFocusedListItem();
		var prevEditIndex = focusedItem.EditIndex;
		if (_index == _showListItemSpan.Index && _showListItemSpan.Index > 0)
		{
			_showListItemSpan = _showListItemSpan.Move(-1);
		}

		var isEditEnd = (prevEditIndex == focusedItem.Value.Length);
		return (prevEditIndex, isEditEnd);
	}

	private void ChildrenOnCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
	{
		if (e.Action == NotifyCollectionChangedAction.Add)
		{
			AddChild(e.NewItems!);
		}
	}
	private IConsoleEditableElement GetFocusedListItem()
	{
		if (_index == -1)
		{
			return new EmptyElement();
		}

		return Children[_index];
	}

	private void JumpDownToItem()
	{
		if (_index == -1)
		{
			return;
		}

		var downInfo = BeforeMoveDown();
		_index = Math.Min(_index + 1, Children.Count - 1);
		AfterMove(downInfo);
	}

	private void JumpUpToItem()
	{
		if (_index == -1)
		{
			return;
		}

		var beforeInfo = BeforeMoveUp();
		_index = Math.Max(_index - 1, 0);
		AfterMove(beforeInfo);
	}

	private void RearrangeChildrenIndex()
	{
		var focusedItem = GetFocusedListItem();
		foreach (var child in Children)
		{
			if (child != focusedItem)
			{
				child.ForceSetEditIndex(focusedItem.EditIndex);
				child.Background = _consoleManager.InputBackgroundColor;
			}
			else
			{
				child.Background = _consoleManager.HighlightBackgroundColor1;
			}
		}
	}

	public void Refresh()
	{
		RearrangeChildrenIndex();
	}
}

