﻿using System;
using System.Linq;

namespace T1.ConsoleUiMixedReality;

public class TextBox : IConsoleEditableElement
{
	private int _editIndex;
	private bool _isSelectedMode;
	private int _startSelectIndex;
	public TextBox()
	{
		Children = new StackChildren(this);
	}

	public event EventHandler<ConsoleElementEvent>? OnHandleEnter;
	public Color BackgroundColor { get; set; } = ConsoleColor.DarkBlue;
	public StackChildren Children { get; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Position CursorPosition
	{
		get
		{
			if (_editIndex < ViewRect.Width)
			{
				return new Position(ViewRect.Left + _editIndex, ViewRect.Top);
			}
			return new Position(ViewRect.Left + ViewRect.Width, ViewRect.Top);
		}
	}

	public object? DataContext { get; set; }
	public Rect DesignRect { get; set; } = new Rect()
	{
		Width = 10,
		Height = 1,
	};

	public int EditIndex
	{
		get => _editIndex;
		set
		{
			if (value >= Value.Length)
			{
				_editIndex = Value.Length - 1;
			}
			_editIndex = value;
		}
	}

	public bool Enabled { get; set; }
	public Color? HighlightBackgroundColor { get; set; }
	public bool IsTab { get; set; } = true;
	public int MaxLength { get; set; } = int.MaxValue;
	public string Name { get; set; } = string.Empty;
	public IConsoleElement? Parent { get; set; }
	public char TypeCharacter { get; set; } = '\0';
	public object? UserObject { get; set; }

	public string Value { get; set; } = String.Empty;

	public Rect ViewRect { get; set; }

	public Character this[Position pos]
	{
		get
		{
			var rect = ViewRect;
			if (!rect.Contain(pos))
			{
				return Character.Empty;
			}

			var contentSpan = GetShowContentSpan(rect);
			var showContent = GetShowContent(contentSpan);

			var x = pos.X - rect.Left;
			var selectedSpan = GetSelectedSpan().Intersect(contentSpan);
			if (!selectedSpan.IsEmpty)
			{
				var selectedValue = GetSelectedValue(selectedSpan);
				if (selectedSpan.Contain(x))
				{
					return new Character(selectedValue[x - selectedSpan.Index], null, HighlightBackgroundColor);
				}
			}

			if (x >= showContent.Length)
			{
				return new Character(' ', null, BackgroundColor);
			}

			if (TypeCharacter != '\0')
			{
				return new Character(TypeCharacter, null, BackgroundColor);
			}

			return new Character(showContent[x], null, BackgroundColor);
		}
	}

	public void ForceSetEditIndex(int index)
	{
		_editIndex = index;
	}

	public Rect GetChildrenRect()
	{
		return ViewRect;
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
		HighlightBackgroundColor ??= consoleManager.HighlightBackgroundColor1;
	}

	public bool OnInput(InputEvent inputEvent)
	{
		var newText = (string?)null;

		if (!_isSelectedMode && inputEvent.HasShift)
		{
			_startSelectIndex = _editIndex;
			_isSelectedMode = true;
		}

		switch (inputEvent.Key)
		{
			case ConsoleKey.Tab:
				Parent?.OnBubbleKeyEvent(this, inputEvent);
				return true;

			case ConsoleKey.LeftArrow:
				_editIndex = Math.Max(0, _editIndex - 1);
				_isSelectedMode = inputEvent.HasShift;
				break;

			case ConsoleKey.RightArrow:
				_isSelectedMode = inputEvent.HasShift;
				if (_editIndex + 1 > Value.Length)
				{
					break;
				}

				_editIndex = Math.Min(Value.Length, _editIndex + 1);
				break;

			case ConsoleKey.UpArrow:
			case ConsoleKey.DownArrow:
				break;

			case ConsoleKey.Backspace:
				_editIndex = Math.Max(0, _editIndex - 1);
				newText = $"{Value.Substring(0, _editIndex)}{Value.SubStr(_editIndex + 1)}";
				_isSelectedMode = false;
				break;

			case ConsoleKey.Delete:
				if (_isSelectedMode)
				{
					var showContentSpan = GetShowContentSpanByView();
					var selectedSpan = GetSelectedSpan();
					var remainingSpans = selectedSpan.NonIntersect(showContentSpan).ToArray();
					newText = string.Join("", remainingSpans.Select(x => Value.Substring(x.Index, x.Length)));
					_editIndex = remainingSpans[0].Index + 1;
				}
				else
				{
					newText = $"{Value.Substring(0, _editIndex)}{Value.SubStr(_editIndex + 1)}";
				}

				_isSelectedMode = false;
				break;

			case ConsoleKey.Home:
				_editIndex = 0;
				break;

			case ConsoleKey.End:
				_editIndex = Value.Length;
				break;

			case ConsoleKey.Enter:
				var consoleElementEvent = new ConsoleElementEvent()
				{
					Element = this,
					InputEvent = inputEvent,
				};
				OnHandleEnter?.Invoke(this, consoleElementEvent);
				Parent?.OnBubbleEvent(this, consoleElementEvent);
				break;

			default:
				if (Value.Length + 1 > MaxLength)
				{
					break;
				}

				var character = inputEvent.Key == ConsoleKey.Enter
					 ? '\n'
					 : inputEvent.KeyChar;
				newText = $"{Value.Substring(0, _editIndex)}{character}{Value.SubStr(_editIndex)}";
				_editIndex = Math.Min(newText.Length, _editIndex + 1);
				_isSelectedMode = false;
				break;
		}

		if (newText != null)
		{
			Value = newText;
		}

		return true;
	}

	public void Refresh()
	{
	}
	private Span GetSelectedSpan()
	{
		if (!_isSelectedMode)
		{
			return Span.Empty;
		}

		var startIndex = Math.Min(_editIndex, _startSelectIndex);
		var endIndex = Math.Max(_editIndex, _startSelectIndex);
		return new Span
		{
			Index = startIndex,
			Length = endIndex - startIndex,
		};
	}

	private string GetSelectedValue(Span selectedSpan)
	{
		return Value.Substring(selectedSpan.Index, selectedSpan.Length);
	}

	private string GetShowContent(Span contentSpan)
	{
		return Value.SubStr(contentSpan.Index, contentSpan.Length);
	}

	private Span GetShowContentSpan(Rect rect)
	{
		var startIndex = _editIndex - rect.Width;
		if (_editIndex < rect.Width)
		{
			startIndex = 0;
		}

		var len = Math.Min(Value.Length, rect.Width);
		return new Span
		{
			Index = startIndex,
			Length = len
		};
	}

	private Span GetShowContentSpanByView()
	{
		var rect = ViewRect.Intersect(ViewRect);
		return GetShowContentSpan(rect);
	}
}