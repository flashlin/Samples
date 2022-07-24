using System.Net.Mime;

namespace GitCli.Models.ConsoleMixedReality;

public interface IConsoleElement
{
    Character this[Position pos] { get; }
    Position CursorPosition { get; }
    Rect ViewRect { get; set; }
    IConsoleElement? Parent { get; set; }
    bool IsTab { get; set; }
    Rect DesignRect { get; set; }
    string Name { get; set; }
    Color BackgroundColor { get; set; }
    StackChildren Children { get; }
    IConsoleManager ConsoleManager { get; set; }
    object? DataContext { get; set; }
    string Value { get; }
    object? UserObject { get; set; }
    bool OnInput(InputEvent inputEvent);
    void OnCreate(Rect parentRect, IConsoleManager consoleManager);
    bool OnBubbleKeyEvent(IConsoleElement element, InputEvent inputEvent);
    void Refresh();
    bool OnBubbleEvent(IConsoleElement element, ConsoleElementEvent evt);
}