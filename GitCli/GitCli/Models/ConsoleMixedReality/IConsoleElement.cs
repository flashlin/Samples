using System.Net.Mime;

namespace GitCli.Models.ConsoleMixedReality;

public interface IConsoleElement
{
    //void Redraw(IConsoleElement control);
    //void Update(IConsoleElement control, Rect rect);
    Func<Rect> GetViewRect { get; }
    Position CursorPosition { get; }
    Character this[Position pos] { get; }
    bool OnInput(InputEvent inputEvent);
}