using System.Net.Mime;

namespace GitCli.Models.ConsoleMixedReality;

public interface IConsoleElement
{
    Character this[Position pos] { get; }
    Position CursorPosition { get; }
    Rect ViewRect { get; set; }
    IConsoleElement? Parent { get; set; }
    bool OnInput(InputEvent inputEvent);
    void OnCreated(IConsoleWriter console);
}