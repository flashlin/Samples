using System.Collections.ObjectModel;
using System.Collections.Specialized;

namespace GitCli.Models.ConsoleMixedReality;

public class StackChildren : ObservableCollection<IConsoleElement>
{
	private int _focusIndex = -1;

	public StackChildren(IConsoleElement parent)
	{
	}

	public IConsoleElement GetFocusedControl()
	{
		if (_focusIndex == -1)
		{
			return new EmptyElement();
		}
		return this[_focusIndex];
	}

	public void JumpDownFocus()
	{
		_focusIndex = Math.Min(_focusIndex + 1, Count - 1);
	}

	public void JumpUpFocus()
	{
		_focusIndex = Math.Min(_focusIndex - 1, 0);
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
}