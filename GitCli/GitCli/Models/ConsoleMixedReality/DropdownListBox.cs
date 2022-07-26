using System.Collections.ObjectModel;
using System.Collections.Specialized;

namespace GitCli.Models.ConsoleMixedReality;

public class DropdownListBox : IConsoleElement
{
	private readonly ListBox _listBox;
	private readonly TextBox _textBox;
	private NotifyCollection<ListItem>? _dataContext;
	private NotifyCollection<ListItem> _list = new();

	private bool _isSelectMode = false;

	public DropdownListBox()
	{
		Children = new StackChildren(this);
		_textBox = new TextBox();
		_listBox = new ListBox(Rect.Empty);
	}

	public Color BackgroundColor { get; set; } = ConsoleColor.Blue;
	public StackChildren Children { get; }
	public IConsoleManager ConsoleManager { get; set; } = EmptyConsoleManager.Default;
	public Position CursorPosition
	{
		get
		{
			if (_isSelectMode)
			{
				return _listBox.CursorPosition;
			}

			return _textBox.CursorPosition;
		}
	}

	public object? DataContext
	{
		get => _dataContext;
		set => SetDataContext(value);
	}

	public Rect DesignRect { get; set; } = new Rect()
	{
		Width = 10,
		Height = 1,
	};

	public bool IsTab { get; set; } = true;
	public string Name { get; set; } = string.Empty;
	public IConsoleElement? Parent { get; set; }
	public object? UserObject { get; set; }
	public string Value { get; set; } = string.Empty;
	public Rect ViewRect { get; set; }


	public Character this[Position pos]
	{
		get
		{
			var y = pos.Y - ViewRect.Top;
			if (y == 0)
			{
				return _textBox[pos];
			}

			if (ConsoleManager.FocusedElement != this)
			{
				return Character.Empty;
			}

			if (!_listBox.ViewRect.Contain(pos))
			{
				return new Character(' ', BackgroundColor);
			}
			return _listBox[pos];
		}
	}

	public TextBox AddItem(ListItem item)
	{
		return _listBox.AddItem(item);
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
		_textBox.Parent = this;
		var textBoxRect = new Rect
		{
			Left = rect.Left,
			Top = rect.Top,
			Width = rect.Width,
			Height = 1,
		};
		_textBox.OnCreate(textBoxRect, consoleManager);

		var childRect = new Rect
		{
			Left = rect.Left,
			Top = rect.Top + 1,
			Width = rect.Width,
			Height = 0,
		};
		_listBox.Parent = this;
		_listBox.Name = "aa";
		_listBox.OnCreate(childRect, consoleManager);
	}

	public bool OnInput(InputEvent inputEvent)
	{
		var flag = false;
		switch (inputEvent.Key)
		{
			case ConsoleKey.UpArrow:
			case ConsoleKey.DownArrow:
				flag = _listBox.OnInput(inputEvent);
				ConsoleManager.FocusedElement = this;
				Refresh();
				return flag;
		}
		flag = _textBox.OnInput(inputEvent);
		Refresh();
		return flag;
	}

	public void Refresh()
	{
		if (ConsoleManager.FocusedElement == this)
		{
			_textBox.ViewRect = new Rect()
			{
				Left = ViewRect.Left,
				Top = ViewRect.Top,
				Width = ViewRect.Width,
				Height = 1,
			};
			var text = _textBox.Value;
			var dataContext = (NotifyCollection<ListItem>?)_dataContext;
			if (dataContext != null)
			{
				var filter = dataContext.ToList().Where(x => x.Title.Contains(text)).ToArray();
				_list.Clear();
				_list.Init(filter);
			}

			var height = Math.Min(10, Math.Min(ViewRect.Height - 1, _list.Count));
			_listBox.ViewRect = new Rect()
			{
				Left = ViewRect.Left,
				Top = ViewRect.Top + 1,
				Width = ViewRect.Width,
				Height = height,
			};
			_listBox.Refresh();
			return;
		}
		_listBox.ViewRect = new Rect()
		{
			Left = ViewRect.Left,
			Top = ViewRect.Top + 1,
			Width = ViewRect.Width,
			Height = 0,
		};
	}

	public void SetDataContext(object? data)
	{
		if (_dataContext != null)
		{
			_dataContext.OnNotify -= OnDataUpdate;
		}
		var dataModel= _dataContext = (NotifyCollection<ListItem>?)data;
		if (_dataContext != null)
		{
			_dataContext.OnNotify += OnDataUpdate;
			OnDataUpdate(data, new NotifyEventArgs<ListItem>());
		}
		if (dataModel != null)
		{
			_list.Init(dataModel.ToList());
			_listBox.DataContext = _list;
		}
	}

	private void OnDataUpdate(object? sender, NotifyEventArgs<ListItem> eventArgs)
	{
		var dataModel = (NotifyCollection<ListItem>)sender!;
		Refresh();
	}
}