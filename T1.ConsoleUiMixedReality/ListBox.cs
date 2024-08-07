﻿using System;
using System.Collections;
using System.Collections.Specialized;
using System.Linq;
using T1.ConsoleUiMixedReality.ModelViewViewmodel;

namespace T1.ConsoleUiMixedReality;

public class ListBox : IConsoleElement
{
	private NotifyCollection<ListItem>? _dataContext;
	private Span _showListSpan = Span.Empty;
	public ListBox(Rect rect)
	{
		DesignRect = rect;
		Children = new StackChildren(this);
		Children.CollectionChanged += ChildrenOnCollectionChanged;
	}

	public Color BackgroundColor { get; set; } = ConsoleColor.Blue;
	public StackChildren Children { get; }
	public IModelCommand? Command { get; set; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Position CursorPosition => Children.FocusedControlOrMe(
		x => x.CursorPosition,
		() => ViewRect.TopLeftCorner);

	public object? DataContext
	{
		get => _dataContext;
		set => SetDataContext(value);
	}

	public Rect DesignRect { get; set; }
	public Color? HighlightBackgroundColor { get; set; }
	public bool IsTab { get; set; } = true;
	public int MaxLength { get; set; } = int.MaxValue;
	public string Name { get; set; } = string.Empty;
	public IConsoleElement? Parent { get; set; }
	public object? UserObject { get; set; }
	public string Value { get; set; }
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

	public void AddElement(IConsoleElement element)
	{
		element.Parent = this;
		element.DesignRect = new Rect()
		{
			Width = DesignRect.Width,
			Height = Math.Max(1, element.DesignRect.Height),
		};
		Children.AddElement(element);
	}

	public TextBox AddItem(ListItem item)
	{
		var textBox = new TextBox
		{
			Parent = this,
			DesignRect = new Rect()
			{
				Width = DesignRect.Width,
				Height = 1,
			},
			Value = item.Title,
			UserObject = item.Value,
		};
		Children.AddElement(textBox);
		return textBox;
	}

	public bool OnBubbleEvent(IConsoleElement element, ConsoleElementEvent evt)
	{
		//this.OnHandleEnter?.Invoke(this, evt);
		Value = Children.GetFocusedControl().Value;
		Command.Raise(evt);
		return true;
	}

	public bool OnBubbleKeyEvent(IConsoleElement element, InputEvent inputEvent)
	{
		return Parent.RaiseOnBubbleKeyEvent(element, inputEvent);
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

	public void Refresh()
	{
		var height = ViewRect.Height < 0 ? 0 : ViewRect.Height;
		_showListSpan = new Span()
		{
			Index = Math.Min(height, _showListSpan.Index),
			Length = ViewRect.Height
		};

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

	public void SetDataContext(object? data)
	{
		if (_dataContext != null)
		{
			_dataContext.OnNotify -= OnDataContext;
		}
		var dataModel = _dataContext = (NotifyCollection<ListItem>?)data;
		if (dataModel != null)
		{
			dataModel.OnNotify += OnDataContext;
			var lastItems = dataModel.ToList();
			OnDataContext(data, new NotifyEventArgs<ListItem>()
			{
				Items = lastItems,
				Status = ChangeStatus.Added,
				LastItems = lastItems,
			});
		}
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

	private void OnDataContext(object? sender, NotifyEventArgs<ListItem> eventArgs)
	{
		Children.Clear();
		var items = eventArgs.LastItems.ToList();
		foreach (var item in items)
		{
			AddItem(item);
		}
		Refresh();
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

	private bool OnMeInputEvent(InputEvent inputEvent)
	{
		OnBubbleKeyEvent(this, inputEvent);
		return true;
	}
}

