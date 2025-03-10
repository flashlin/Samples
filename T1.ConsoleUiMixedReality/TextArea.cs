﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace T1.ConsoleUiMixedReality;

public class TextArea : IConsoleElement
{
	private int _editIndex;
	private bool _isSelectedMode;
	private int _startSelectIndex;

	public TextArea(Rect rect)
	{
		Children = new StackChildren(this);
		DesignRect = rect;
	}

	public Color Background { get; set; } = ConsoleColor.DarkBlue;
	public Color BackgroundColor { get; set; } = ConsoleColor.DarkBlue;

	public StackChildren Children { get; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Position CursorPosition
	{
		get
		{
			var y = _editIndex / ViewRect.Width;
			var x = _editIndex % ViewRect.Width;
			y = Math.Min(y, ViewRect.Height - 1);
			return new Position(ViewRect.Left + x, ViewRect.Top + y);
		}
	}

	public object? DataContext { get; set; }
	public Rect DesignRect { get; set; }
	public int EditIndex => _editIndex;
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

			var contentSpans = GetShowContentSpanList(rect);
			var selectedSpans = GetSelectedSpans(contentSpans).ToList();

			var offsetX = pos.X - rect.Left;
			var index = (pos.Y - rect.Top) * rect.Width + offsetX;
			index += contentSpans.FirstOrDefault().Index;
			foreach (var (selectedSpan, contentSpan) in selectedSpans.Zip(contentSpans, (x, y) => (x, y)))
			{
				if (selectedSpan.Contain(index))
				{
					var contentValue = Value.SubShowStr(contentSpan.Index, contentSpan.Length);
					var ch = contentValue.SubShowStr(offsetX, 1)[0];
					return new Character(ch, null, ConsoleColor.DarkGray);
				}
			}

			foreach (var span in contentSpans)
			{
				if (span.Contain(index))
				{
					var contentValue = Value.SubShowStr(span.Index, span.Length);
					var ch = contentValue.SubShowStr(offsetX, 1)[0];
					return new Character(ch, null, Background);
				}
			}

			return new Character(' ', null, Background);
		}
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

	public void OnCreate(Rect ofSize, IConsoleManager consoleManager)
	{
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
				_isSelectedMode = inputEvent.HasShift;
				if ((_editIndex - ViewRect.Width) >= 0)
				{
					_editIndex -= ViewRect.Width;
				}
				break;
			case ConsoleKey.DownArrow:
				_isSelectedMode = inputEvent.HasShift;
				if ((_editIndex + ViewRect.Width) < Value.Length)
				{
					_editIndex += ViewRect.Width;
				}
				break;

			case ConsoleKey.Backspace:
				_editIndex = Math.Max(0, _editIndex - 1);
				newText = $"{Value.Substring(0, _editIndex)}{Value.SubStr(_editIndex + 1)}";
				_isSelectedMode = false;
				break;

			case ConsoleKey.Delete:
				if (_isSelectedMode)
				{
					var showContentSpan = new Span
					{
						Index = 0,
						Length = Value.Length
					};
					var selectedSpan = GetSelectedSpan();
					var remainingSpans = selectedSpan.NonIntersect(showContentSpan).ToArray();
					newText = string.Join("", remainingSpans.Select(x => Value.Substring(x.Index, x.Length)));
					_editIndex = selectedSpan.Index;
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
	public void SetDataContext(object dataModel)
	{
	}

	private IEnumerable<Span> GetContentSpans(Rect rect)
	{
		var editIndex = 0;
		var valueLength = Value.Length + 1;
		while (valueLength > 0)
		{
			yield return new Span
			{
				Index = editIndex,
				Length = Math.Min(valueLength, rect.Width)
			};
			valueLength -= rect.Width;
			editIndex += rect.Width;
		}
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

	private IEnumerable<Span> GetSelectedSpans(List<Span> contentSpans)
	{
		if (!_isSelectedMode)
		{
			yield break;
		}

		var selectedSpan = GetSelectedSpan();
		foreach (var contentSpan in contentSpans)
		{
			var span = contentSpan.Intersect(selectedSpan);
			if (span.IsEmpty)
			{
				yield return new Span
				{
					Index = contentSpan.Index,
					Length = -contentSpan.Length,
				};
			}
			else
			{
				yield return span;
			}
		}
	}

	private string GetSelectedValue(Span selectedSpan)
	{
		return Value.Substring(selectedSpan.Index, selectedSpan.Length);
	}

	private string GetShowContent(Span contentSpan)
	{
		return Value.Substring(contentSpan.Index, contentSpan.Length);
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
		return GetShowContentSpan(ViewRect);
	}
	private List<Span> GetShowContentSpanList(Rect rect)
	{
		var editHeight = _editIndex / rect.Width;
		var startHeight = Math.Max(editHeight - rect.Height + 1, 0);
		var list = GetContentSpans(rect)
			 .Skip(startHeight)
			 .Take(rect.Height)
			 .ToList();
		return list;
	}
}