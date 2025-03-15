namespace VimSharpLib;

public static class ColoredCharArrayExtensions
{
    public static void SetText(this ColoredChar[,] chars, int x, int y, ConsoleText line)
    {
        for (var i = 0; i < line.Width; i++)
        {
            var coloredChar = line.Chars[i];
            chars[x + i, y] = coloredChar;
        }
    }
    
    public static void Set(this ColoredChar[,] chars, int x, int y, ColoredChar character)
    {
        if(x < 0 || x >= chars.GetLength(0) || y < 0 || y >= chars.GetLength(1))
        {
            return;
        }
        chars[x, y] = character;
    }
}