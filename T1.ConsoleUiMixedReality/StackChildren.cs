﻿using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Linq;

namespace T1.ConsoleUiMixedReality;

public class StackChildren : ObservableCollection<IConsoleElement>
{
	private int _focusIndex = -1;
	private readonly IConsoleElement _parent;

	public StackChildren(IConsoleElement parent)
	{
		_parent = parent;
	}

	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public int FocusIndex => _focusIndex;

	public IConsoleElement GetFocusedControl()
	{
		if (_focusIndex == -1)
		{
			return EmptyElement.Default;
		}
		if (_focusIndex > Count)
		{
			_focusIndex = Count - 1;
		}
		return this[_focusIndex];
	}

	public T FocusedControlOrMe<T>(Func<IConsoleElement, T> action, Func<T> parentAction)
	{
		if (_focusIndex == -1 || Count == 0)
		{
			return parentAction();
		}
		return action(this[_focusIndex]);
	}

	public bool JumpDownFocus()
	{
		if (_focusIndex + 1 >= Count)
		{
			return false;
		}
		_focusIndex = Math.Min(_focusIndex + 1, Count - 1);
		return true;
	}

	public bool JumpUpFocus()
	{
		if (_focusIndex - 1 < 0)
		{
			return false;
		}
		_focusIndex = Math.Max(_focusIndex - 1, 0);
		return true;
	}

	protected override void OnCollectionChanged(NotifyCollectionChangedEventArgs e)
	{
		if (e.Action == NotifyCollectionChangedAction.Add)
		{
			_focusIndex = Math.Max(_focusIndex, 0);
		}
		base.OnCollectionChanged(e);
	}

	public Character GetContent(Position pos)
	{
		foreach (var child in this)
		{
			var ch = child[pos];
			if (ch != Character.Empty)
			{
				return ch;
			}
		}

		return this.GetFocusedControl()[pos];
	}

	public void ForEachIndex(Action<IConsoleElement, int> eachAction)
	{
		foreach (var child in this.Select((val, idx) => (val, idx)))
		{
			eachAction(child.val, child.idx);
		}
	}

	public void AddElement(IConsoleElement element)
	{
		this.Add(element);
		ConsoleManager.SetFocusElementOrChild(_parent, element);
	}
}
