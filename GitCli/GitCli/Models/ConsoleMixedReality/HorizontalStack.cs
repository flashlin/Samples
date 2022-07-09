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
}


public class HorizontalStack : IConsoleElement
{
	private int _focusIndex = -1;

	public HorizontalStack(Rect rect)
	{
		ViewRect = rect;
		Children = new StackChildren(this);
	}

	public StackChildren Children { get; private set; }

	public Position CursorPosition => Children.GetFocusedControl().CursorPosition;

	public bool IsTab { get; set; }
	public IConsoleElement? Parent { get; set; }
	public Rect ViewRect { get; set; } = Rect.Empty;
	public Character this[Position pos]
	{
		get
		{
			if (!ViewRect.Contain(pos))
			{
				return Character.Empty;
			}
			return Children.GetFocusedControl()[pos];
		}
	}

	public void OnBubbleEvent(IConsoleElement element, InputEvent inputEvent)
	{
		if (inputEvent.Key == ConsoleKey.Tab && inputEvent.HasShift)
		{
			Children.JumpUpFocus();
			return;
		}

		if (inputEvent.Key == ConsoleKey.Tab)
		{
			Children.JumpDownFocus();
			return;
		}

		if (inputEvent.HasControl && inputEvent.Key == ConsoleKey.UpArrow)
		{
			if (_focusIndex != -1)
			{
				Children.JumpUpFocus();
				return;
			}

			Parent?.OnBubbleEvent(this, inputEvent);
			return;
		}

		if ((inputEvent.HasControl && inputEvent.Key == ConsoleKey.DownArrow) || inputEvent.Key == ConsoleKey.Enter)
		{
			if (_focusIndex != -1)
			{
				Children.JumpDownFocus();
				return;
			}
		}

		Parent?.OnBubbleEvent(this, inputEvent);
	}

	public void OnCreate(IConsoleManager manager)
	{
		var viewRect = ViewRect = ViewRect.Init(() => Rect.OfSize(manager.Console.GetSize()));
		var left = ViewRect.Left;
		foreach (var (child, idx) in Children.Select((val, idx) => (val, idx)))
		{
			if (idx == 0)
			{
				_focusIndex = 0;
				left = viewRect.Left + child.ViewRect.Left;
			}
			child.Parent = this;
			child.ViewRect = new Rect
			{
				Left = left,
				Top = viewRect.Top + child.ViewRect.Top,
				Width = child.ViewRect.Width,
				Height = child.ViewRect.Height,
			};
			child.OnCreate(manager);
			left += child.ViewRect.Width;
		}
	}

	public bool OnInput(InputEvent inputEvent)
	{
		return Children.GetFocusedControl().OnInput(inputEvent);
	}
}