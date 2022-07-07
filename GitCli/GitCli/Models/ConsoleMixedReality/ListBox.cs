using System.Collections;
using System.Collections.ObjectModel;
using System.Collections.Specialized;

namespace GitCli.Models.ConsoleMixedReality;

public class ListBox : IConsoleElement
{
    private int _editIndex;
    private int _index = -1;
    private int _startSelectIndex;
    private bool _isSelectedMode;
    private int _maxLength;

    private Span _showListItemSpan = Span.Empty;
    private IConsoleWriter _console;

    public ListBox(Rect rect)
    {
        ViewRect = rect;
        Children.CollectionChanged += ChildrenOnCollectionChanged;
    }

    private void ChildrenOnCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (e.Action == NotifyCollectionChangedAction.Add)
        {
            AddChild(e.NewItems!);
        }
    }

    private void AddChild(IList newItems)
    {
        foreach (IConsoleElement item in newItems)
        {
            item.Parent = this;
        }
    }

    public IConsoleElement? Parent { get; set; }

    public ObservableCollection<TextBox> Children { get; } = new();

    public Color BackgroundColor { get; set; } = ConsoleColor.Blue;

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

    public Rect ViewRect { get; set; }
    public int MaxLength { get; set; } = int.MaxValue;

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
            //recalute item view
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

    public bool OnInput(InputEvent inputEvent)
    {
        var focusedItem = (IConsoleEditableElement?) null;
        var prevEditIndex = 0;
        var isEditEnd = false;
        switch (inputEvent.Key)
        {
            case ConsoleKey.LeftArrow:
                //_editIndex = Math.Max(0, _editIndex - 1);
                GetFocusedListItem().OnInput(inputEvent);
                return true;

            case ConsoleKey.RightArrow:
                //if (_editIndex + 1 > _maxLength)
                //{
                //	break;
                //}
                //_editIndex = Math.Min(_maxLength, _editIndex + 1);
                GetFocusedListItem().OnInput(inputEvent);
                return true;

            case ConsoleKey.UpArrow:
                if (_index == -1)
                {
                    break;
                }

                var beforeInfo = BeforeMoveUp();
                _index = Math.Max(_index - 1, 0);
                AfterMove(beforeInfo);
                break;
            case ConsoleKey.DownArrow:
                if (_index == -1)
                {
                    break;
                }

                var downInfo = BeforeMoveDown();
                _index = Math.Min(_index + 1, Children.Count - 1);
                AfterMove(downInfo);
                break;

            // case ConsoleKey.Home:
            //     _editIndex = 0;
            //     break;
            //
            // case ConsoleKey.End:
            //     _editIndex = Value.Length;
            //     break;

            case ConsoleKey.Enter:
                break;
        }

        return true;
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

    public void OnCreated(IConsoleWriter console)
    {
        _console = console;
        var y = ViewRect.Top;
        foreach (var child in Children)
        {
            _index = 0;
            child.Parent = this;
            child.ViewRect = new Rect()
            {
                Left = ViewRect.Left,
                Top = y,
                Width = ViewRect.Width,
                Height = 1,
            };
            child.OnCreated(console);
            _maxLength = Math.Max(_maxLength, child.Value.Length);
            y += 1;
        }

        _showListItemSpan = new Span()
        {
            Index = 0,
            Length = ViewRect.Height
        };
    }

    public void OnBubbleEvent(InputEvent inputEvent)
    {
    }

    private IConsoleEditableElement GetFocusedListItem()
    {
        if (_index == -1)
        {
            return new EmptyElement();
        }

        return Children[_index];
    }
}