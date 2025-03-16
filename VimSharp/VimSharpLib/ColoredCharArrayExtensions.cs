namespace VimSharpLib;

public static class ColoredCharArrayExtensions
{
    public static void SetText(this ColoredChar[,] chars, int x, int y, ConsoleText line)
    {
        for (int i = 0; i < line.Width; i++)
        {
            if (x + i < chars.GetLength(1))
            {
                chars[y, x + i] = line.Chars[i];
            }
        }
    }
    
    public static void Set(this ColoredChar[,] chars, int x, int y, ColoredChar c)
    {
        chars[y, x] = c;
    }
}