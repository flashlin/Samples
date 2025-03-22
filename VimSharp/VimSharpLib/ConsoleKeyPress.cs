namespace VimSharpLib;

public static class ConsoleKeyPress
{
    public static ConsoleKeyInfo i = new('i', ConsoleKey.I, false, false, false);
    public static ConsoleKeyInfo DownArrow = new('\0', ConsoleKey.DownArrow, false, false, false);
    public static ConsoleKeyInfo ArrowUp = new('\0', ConsoleKey.UpArrow, false, false, false);
    public static ConsoleKeyInfo LeftArrow = new('\0', ConsoleKey.LeftArrow, false, false, false);
    public static ConsoleKeyInfo RightArrow = new('\0', ConsoleKey.RightArrow, false, false, false);
    public static ConsoleKeyInfo Money = new('$', ConsoleKey.D4, true, false, false);
    public static ConsoleKeyInfo Enter = new('\n', ConsoleKey.Enter, false, false, false);
}