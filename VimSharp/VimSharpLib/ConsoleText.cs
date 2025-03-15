using System.Text;

namespace VimSharpLib;

public class ConsoleText
{
    public ColoredChar[] Chars { get; set; } = [];

    public int Width
    {
        get => Chars.Length;
    }

    public void SetWidth(int width)
    {
        if (Chars.Length < width)
        {
            var newChars = new ColoredChar[width];
            Array.Copy(Chars, newChars, Chars.Length);
            for (var i = Chars.Length; i < width; i++)
            {
                newChars[i] = new ColoredChar(' ');
            }
            Chars = newChars;
            return;
        }
        if (Chars.Length > width)
        {
            var newChars = new ColoredChar[width];
            Array.Copy(Chars, newChars, width);
            Chars = newChars;
        }
    }

    public void SetText(int x, string text)
    {
        SetWidth(text.GetTextWidth());
        for (var i = 0; i < text.Length; i++)
        {
            var coloredChar = new ColoredChar(text[i]);
            Chars[x + i] = coloredChar;
            
            if (coloredChar.Char > 127 && x + i + 1 < Chars.Length)
            {
                Chars[x + i + 1] = ColoredChar.None;
            }
        }
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        foreach (var c in Chars)
        {
            if (c.Char == '\0')
            {
                continue;
            }
            sb.Append(c.Char);
        }
        return sb.ToString();
    }

    public string ToAnsiString()
    {
        var sb = new StringBuilder();
        foreach (var c in Chars)
        {
            if (c.Char == '\0')
            {
                continue;
            }
            sb.Append(c.ToAnsiString());
        }
        return sb.ToString();
    }

    public void SetColor(ConsoleColor foregroundColor, ConsoleColor backgroundColor)
    {
        foreach (var c in Chars)
        {
            if( c == ColoredChar.None)
            {
                continue;
            }
            if( c == ColoredChar.Empty)
            {
                continue;
            }
            c.ForegroundColor = foregroundColor;
            c.BackgroundColor = backgroundColor;
        }
    }
}