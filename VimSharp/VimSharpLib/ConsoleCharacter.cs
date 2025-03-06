namespace VimSharpLib;

public class ConsoleCharacter
{
    public static ConsoleCharacter Empty = new ConsoleCharacter
    {
        Value = '\0',
        Color = ConsoleColor.Black,
        BackgroundColor = ConsoleColor.Black
    };
    public char Value { get; set; }
    public ConsoleColor Color { get; set; }
    public ConsoleColor BackgroundColor { get; set; }
}