namespace VimSharpLib;

public class ColoredChar
{
    /// <summary>
    /// 空字符（黑底白字的空格）
    /// </summary>
    public static readonly ColoredChar Empty = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.DarkGray);
    public static readonly ColoredChar None = new ColoredChar('\0', ConsoleColor.White, ConsoleColor.DarkGray);
        

    public char Char { get; set; }
    public ConsoleColor ForegroundColor { get; set; }
    public ConsoleColor BackgroundColor { get; set; }
    
    public ColoredChar(char c, 
        ConsoleColor foregroundColor = ConsoleColor.White, 
        ConsoleColor backgroundColor = ConsoleColor.DarkGray)
    {
        Char = c;
        ForegroundColor = foregroundColor;
        BackgroundColor = backgroundColor;
    }
    
    // 轉換為 ANSI 控制碼字串
    public string ToAnsiString()
    {
        return $"\u001b[38;5;{(int)ForegroundColor}m\u001b[48;5;{(int)BackgroundColor}m{Char}\u001b[0m";
    }
    
    // 隱式轉換為 char
    public static implicit operator char(ColoredChar coloredChar) => coloredChar.Char;
    
    // 隱式轉換自 char
    public static implicit operator ColoredChar(char c) => new ColoredChar(c);
} 