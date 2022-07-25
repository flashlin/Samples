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
			Height = rect.Height - 1,
		};
		_listBox.Parent = this;
		_listBox.OnCreate(childRect, consoleManager);
	}

	public bool OnInput(InputEvent inputEvent)
	{
		switch (inputEvent.Key)
		{
			case ConsoleKey.UpArrow:
			case ConsoleKey.DownArrow:
				var flag = _listBox.OnInput(inputEvent);
				ConsoleManager.FocusedElement = this;
				return flag;
		}
		return _textBox.OnInput(inputEvent);
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
			_listBox.ViewRect = new Rect()
			{
				Left = ViewRect.Left,
				Top = ViewRect.Top + 1,
				Width = ViewRect.Width,
				Height = 10,
			};
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

	public void SetDataContext(object? dataModel)
	{
		if (_dataContext != null)
		{
			_dataContext.OnNotify -= OnDataContext;
		}
		_dataContext = (NotifyCollection<ListItem>?)dataModel;
		if (_dataContext != null)
		{
			_dataContext.OnNotify += OnDataContext;
			OnDataContext(dataModel, new NotifyEventArgs<ListItem>());
		}
	}

	private void OnDataContext(object? sender, NotifyEventArgs<ListItem> eventArgs)
	{
		//var dataModel = (NotifyCollection<ListItem>)sender!;
		//_listBox.DataContext = dataModel;
		//Refresh();
		
		var dataModel = (NotifyCollection<ListItem>)sender!;
		_list.Init(dataModel.ToList());
		_dataContext = dataModel;
		_listBox.DataContext = _list;
		Refresh();
	}
}