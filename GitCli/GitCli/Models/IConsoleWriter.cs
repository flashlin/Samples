namespace GitCli.Models;

public interface IConsoleWriter
{
    void SetForegroundColor(ConsoleColor color);
    void SetBackgroundColor(ConsoleColor color);
    void Write(string text);
    void WriteLine(string text);
    void ResetWriteColor();
    void ResetColor();
}