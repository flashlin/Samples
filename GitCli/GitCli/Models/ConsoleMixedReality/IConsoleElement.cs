using System.ComponentModel;
using System.Net.Mime;

namespace GitCli.Models.ConsoleMixedReality;

public interface IConsoleElement
{
    Character this[Position pos] { get; }
    Position CursorPosition { get; }
    Rect ViewRect { get; set; }
    IConsoleElement? Parent { get; set; }
    bool OnInput(InputEvent inputEvent);
    void OnCreate(IConsoleManager manager);
    void OnBubbleEvent(InputEvent inputEvent);
}

public interface IConsoleEditableElement : IConsoleElement
{
    int EditIndex { get; set; }
    string Value { get; }
    void ForceSetEditIndex(int index);
}

public interface IRaisePropertyChanged : INotifyPropertyChanged
{
    void RaisePropertyChanged(string propertyName, object? value);
}

public class ComponentProperty<TValue, TOwner>
    where TOwner : IRaisePropertyChanged
{
    private TValue? _value;
    private readonly TOwner _owner;
    private readonly string _propertyName;

    public ComponentProperty(TOwner owner, string propertyName)
    {
        _propertyName = propertyName;
        _owner = owner;
    }

    public TValue? Value
    {
        get => _value;
        set
        {
            if (EqualityComparer<TValue>.Default.Equals(_value, value))
            {
                return;
            }
            _value = value;
            _owner.RaisePropertyChanged(_propertyName, value);
        }
    }
}