namespace GitCli.Models.ConsoleMixedReality;

public interface IConsoleWriter
{
    void SetForegroundColor(ConsoleColor color);
    void SetBackgroundColor(ConsoleColor color);
    void ResetWriteColor();
    void ResetColor();
    Size GetSize();
    void Clear();
    void HideCursor();
    void ShowCursor();
    void Initialize();
    void Write(Position position, Character character);
    ConsoleInputObserver KeyEvents { get; }
    bool IsInsertMode { get; set; }
    void SetCursorPosition(Position position);
    InputEvent ReadKey();
}

