namespace VimSharp
{
    public interface IConsoleDevice
    {
        int WindowWidth { get; }
        int WindowHeight { get; }
        void SetCursorPosition(int left, int top);
        void Write(string value);
        ConsoleKeyInfo ReadKey(bool intercept);
    }
} 