namespace VimSharpLib;

public interface IVimMode
{
    void WaitForInput();
    VimEditor Instance { get; }
    void PressKey(ConsoleKey key);
} 