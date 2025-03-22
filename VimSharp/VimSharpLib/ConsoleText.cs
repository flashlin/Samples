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
                newChars[i] = ColoredChar.Empty;
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

    /// <summary>
    /// 設置文本內容
    /// </summary>
    /// <param name="x">起始 X 位置</param>
    /// <param name="text">文本內容</param>
    public void SetText(int x, string text)
    {
        int width = x + text.GetStringDisplayWidth();
        SetWidth(width);
        int pos = x;
        for (var i = 0; i < text.Length; i++)
        {
            var coloredChar = new ColoredChar(text[i]);
            Chars[pos] = coloredChar;
            if (coloredChar.Char > 127)
            {
                Chars[pos + 1] = ColoredChar.None;
                pos += 2; // 中文字符佔用兩個位置
            }
            else
            {
                pos += 1; // ASCII 字符佔用一個位置
            }
        }
    }

    public override string ToString()
    {
        return Chars.ToText();
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

    public int FindLastCharIndex()
    {
        var lastCharIndex = -1;
        for (var i = Width - 1; i >= 0; i--)
        {
            var currentChar = Chars[i].Char;
            if (currentChar != ' ' && currentChar != '\0')
            {
                lastCharIndex = i;
                break;
            }
        }
        return lastCharIndex;
    }

    public string GetText(int offset)
    {
        var subChars = Chars.Skip(offset).ToArray();
        return subChars.ToText();
    }

    public string Substring(int offset, int length)
    {
        var subChars = Chars.Skip(offset).Take(length).ToArray();
        return subChars.ToText();
    }

    public void Remove(int textX)
    {
        if (textX == 0)
        {
            Chars = [];
            return;
        }
        var width = Width - textX;
        var newChars = new ColoredChar[width];
        Array.Copy(Chars, 0, newChars, 0, width);
        Chars = newChars;
    }
}