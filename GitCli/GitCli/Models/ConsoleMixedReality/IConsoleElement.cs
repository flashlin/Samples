using System.Net.Mime;

namespace GitCli.Models.ConsoleMixedReality;

public interface IConsoleElement
{
    IConsoleWriter VConsole { get; }
    Character this[Position pos] { get; }
    Position CursorPosition { get; }
    Rect ViewRect { get; set; }
    bool OnInput(InputEvent inputEvent);
    void OnCreated();
    internal void SetConsoleInstance(IConsoleWriter console);
}