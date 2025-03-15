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

    public int ToAnsiForegroundColor(ConsoleColor color)
    {
        return color switch
        {
            ConsoleColor.Black => 30,
            ConsoleColor.DarkBlue => 34,
            ConsoleColor.DarkGreen => 32, 
            ConsoleColor.DarkCyan => 36,
            ConsoleColor.DarkRed => 31,
            ConsoleColor.DarkMagenta => 35,
            ConsoleColor.DarkYellow => 33,
            ConsoleColor.Gray => 37,
            ConsoleColor.DarkGray => 90,
            ConsoleColor.Blue => 94,
            ConsoleColor.Green => 92,
            ConsoleColor.Cyan => 96,
            ConsoleColor.Red => 91,
            ConsoleColor.Magenta => 95,
            ConsoleColor.Yellow => 93,
            ConsoleColor.White => 97,
            _ => 37 // 默認為灰色
        };
    }

    public int ToAnsiBackgroundColor(ConsoleColor color)
    {
        return color switch
        {
            ConsoleColor.Black => 40,
            ConsoleColor.DarkBlue => 44,
            ConsoleColor.DarkGreen => 42, 
            ConsoleColor.DarkCyan => 46,
            ConsoleColor.DarkRed => 41,
            ConsoleColor.DarkMagenta => 45,
            ConsoleColor.DarkYellow => 43,
            ConsoleColor.Gray => 47,
            ConsoleColor.DarkGray => 100,
            ConsoleColor.Blue => 104,
            ConsoleColor.Green => 102,
            ConsoleColor.Cyan => 106,
            ConsoleColor.Red => 101,
            ConsoleColor.Magenta => 105,
            ConsoleColor.Yellow => 103,
            ConsoleColor.White => 107,
            _ => 47 // 默認為灰色背景
        };
    }

    // 轉換為 ANSI 控制碼字串
    public string ToAnsiString()
    {
        var foreground = ToAnsiForegroundColor(ForegroundColor);
        var background = ToAnsiBackgroundColor(BackgroundColor);
        return $"\u001b[{foreground}m\u001b[{background}m{Char}\u001b[0m";
    }
    
    // 隱式轉換為 char
    public static implicit operator char(ColoredChar coloredChar) => coloredChar.Char;
    
    // 隱式轉換自 char
    public static implicit operator ColoredChar(char c) => new ColoredChar(c);
} 