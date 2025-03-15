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
}