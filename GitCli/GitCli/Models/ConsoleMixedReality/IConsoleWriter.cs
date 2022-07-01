namespace GitCli.Models.ConsoleMixedReality;

public interface IConsoleWriter
{
    void SetForegroundColor(ConsoleColor color);
    void SetBackgroundColor(ConsoleColor color);
    void Write(string text);
    void WriteLine(string text);
    void ResetWriteColor();
    void ResetColor();
    Size GetSize();
    void Clear();
    void HideCursor();
    void ShowCursor();
    void Initialize();
    void Write(Position position, Character character);
    ConsoleInputObserver KeyEvents { get; }
    void ReadKey();
}

