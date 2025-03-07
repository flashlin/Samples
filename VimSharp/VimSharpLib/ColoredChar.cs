namespace VimSharpLib;

public struct ColoredChar
{
    public char Char { get; set; }
    public ConsoleColor Foreground { get; set; }
    public ConsoleColor Background { get; set; }
    
    public ColoredChar(char c, ConsoleColor foreground = ConsoleColor.White, ConsoleColor background = ConsoleColor.Black)
    {
        Char = c;
        Foreground = foreground;
        Background = background;
    }
    
    // 轉換為 ANSI 控制碼字串
    public string ToAnsiString()
    {
        return $"\u001b[38;5;{(int)Foreground}m\u001b[48;5;{(int)Background}m{Char}\u001b[0m";
    }
    
    // 隱式轉換為 char
    public static implicit operator char(ColoredChar coloredChar) => coloredChar.Char;
    
    // 隱式轉換自 char
    public static implicit operator ColoredChar(char c) => new ColoredChar(c);
} 