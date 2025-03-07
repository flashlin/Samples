using System.Text;

namespace VimSharpLib;

public class ConsoleText
{
    public ColoredChar[] Chars { get; set; } = [];
    
    public int Width
    {
        get => Chars.Length;
        set
        {
            if (Chars.Length < value)
            {
                var newChars = new ColoredChar[value];
                Array.Copy(Chars, newChars, Chars.Length);
                for (var i = Chars.Length; i < value; i++)
                {
                    newChars[i] = new ColoredChar(' ');
                }
                Chars = newChars;
                return;
            }
            if (Chars.Length > value)
            {
                var newChars = new ColoredChar[value];
                Array.Copy(Chars, newChars, value);
                Chars = newChars;
            }
        }
    }

    public void SetText(int x, string text)
    {
        Width = text.Length + x;
        for (var i = 0; i < text.Length; i++)
        {
            Chars[x + i] = new ColoredChar(text[i]);
        }
    }
    
    // 設置帶顏色的文字
    public void SetColoredText(int x, string text, ConsoleColor foreground, ConsoleColor background)
    {
        Width = text.Length + x;
        for (var i = 0; i < text.Length; i++)
        {
            Chars[x + i] = new ColoredChar(text[i], foreground, background);
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
    
    // 獲取帶顏色的字串表示
    public string ToColoredString()
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
}