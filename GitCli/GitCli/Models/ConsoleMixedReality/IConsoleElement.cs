using System.Net.Mime;

namespace GitCli.Models.ConsoleMixedReality;

public interface IConsoleElement
{
    //void Redraw(IConsoleElement control);
    //void Update(IConsoleElement control, Rect rect);
    Func<Rect> GetViewRect { get; }
    Character this[Position pos] { get; }
    bool OnInput(InputEvent inputEvent);
}

public class ConsoleTextBox : IConsoleElement
{
    private int _caretStart;
    private int _caretEnd;
    private int _editIndex;
    private readonly IConsoleWriter _console;
    private FreezeLock _freezeLock;

    public ConsoleTextBox(IConsoleWriter console, FreezeLock freezeLock)
    {
        _console = console;
        _freezeLock = freezeLock;
        GetViewRect = () => Rect.OfSize(_console.GetSize());
    }

    public string Value { get; set; } = String.Empty;
    public int MaxLength { get; set; } = int.MaxValue;
    public Rect EditRect { get; set; }
    public Func<Rect> GetViewRect { get; set; }
    public Character this[Position pos] => throw new NotImplementedException();

    public bool OnInput(InputEvent inputEvent)
    {
        var rect = EditRect.Intersect(GetViewRect());
        var newText = String.Empty;
        switch (inputEvent.Key)
        {
            case ConsoleKey.LeftArrow:
                _editIndex = _caretStart = _caretEnd = Math.Max(0, _caretStart - 1);
                break;
            case ConsoleKey.RightArrow:
                _caretEnd = Math.Min(_caretEnd + 1, Value.Length - 1);
                _caretStart = Math.Min(_caretStart + 1, _caretEnd - rect.Width - 1);
                _editIndex = Math.Max(Value.Length - 1, _editIndex + 1);
                break;
            // case ConsoleKey.Delete when CaretStart != CaretEnd:
            // case ConsoleKey.Backspace when CaretStart != CaretEnd:
            //  newText = $"{Value.Substring(_editIndex, CaretEnd)}";
            //  break;
            // case ConsoleKey key when char.IsControl(inputEvent.Key.KeyChar) && inputEvent.Key.Key != ConsoleKey.Enter:
            //     return;
            default:
                var character = inputEvent.Key == ConsoleKey.Enter
                    ? '\n'
                    : inputEvent.KeyChar;
                newText =
                    $"{Value.Substring(0, _caretStart)}{character}{Value.Substring(_caretEnd)}";
                _editIndex = _caretStart = _caretEnd = _caretStart + 1;
                break;
        }

        if (newText != String.Empty)
        {
            Value = newText;
        }
        return true;
    }
}