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

    public int ToAnsiColor(ConsoleColor color)
    {
        return color switch
        {
            ConsoleColor.Black => 0,
            ConsoleColor.DarkBlue => 4,
            ConsoleColor.DarkGreen => 2, 
            ConsoleColor.DarkCyan => 6,
            ConsoleColor.DarkRed => 1,
            ConsoleColor.DarkMagenta => 5,
            ConsoleColor.DarkYellow => 3,
            ConsoleColor.Gray => 7,
            ConsoleColor.DarkGray => 8,
            ConsoleColor.Blue => 12,
            ConsoleColor.Green => 10,
            ConsoleColor.Cyan => 14,
            ConsoleColor.Red => 9,
            ConsoleColor.Magenta => 13,
            ConsoleColor.Yellow => 11,
            ConsoleColor.White => 15,
            _ => 7 // 默認為灰色
        };
    }
    
    // 轉換為 ANSI 控制碼字串
    public string ToAnsiString()
    {
        var foreground = ToAnsiColor(ForegroundColor);
        var background = ToAnsiColor(BackgroundColor);
        return $"\u001b[38;5;{foreground}m\u001b[48;5;{background}m{Char}\u001b[0m";
    }
    
    // 隱式轉換為 char
    public static implicit operator char(ColoredChar coloredChar) => coloredChar.Char;
    
    // 隱式轉換自 char
    public static implicit operator ColoredChar(char c) => new ColoredChar(c);
} 