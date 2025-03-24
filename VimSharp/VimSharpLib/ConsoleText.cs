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
        var width = x + text.GetStringDisplayWidth();
        SetWidth(width);
        var pos = x;
        foreach (var t in text)
        {
            var coloredChar = new ColoredChar(t);
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
            if (currentChar != ' ' && currentChar != '\0' && currentChar != '\n')
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
    
    public ColoredChar[] GetChars(int offset, int? length=null)
    {
        var subChars = Chars.Skip(offset).ToArray();
        if (length != null)
        {
            length = Math.Min(length.Value, subChars.Length);
            subChars = subChars.Take(length.Value).ToArray();
        }
        return subChars;
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
        var reverseWidth = textX;
        var newChars = new ColoredChar[reverseWidth];
        Array.Copy(Chars, 0, newChars, 0, reverseWidth);
        Chars = newChars;
    }

    public void InsertText(int offset, string text)
    {
        var textWidth = text.GetStringDisplayWidth();
        var width = Width + textWidth;
        var newChars = new ColoredChar[width];
        Array.Copy(Chars, 0, newChars, 0, offset);
        for (var i = 0; i < text.Length; i++)
        {
            var coloredChar = new ColoredChar(text[i]);
            newChars[offset + i] = coloredChar;
            if (coloredChar.Char > 127)
            {
                newChars[offset + i + 1] = ColoredChar.None;
                i++;
            }
        }
        Array.Copy(Chars, offset, newChars, offset + textWidth, Width - offset);
        Chars = newChars;
    }

    /// <summary>
    /// 從指定位置開始尋找下一個單詞的起始位置
    /// </summary>
    /// <param name="offset">開始搜尋的位置</param>
    /// <returns>下一個單詞的起始位置，如果找不到則返回 -1</returns>
    public int IndexOfNextWord(int offset)
    {
        var wordsIndexList = Chars.QueryWordsIndexList();
        var nextWordIndexList = wordsIndexList.Where(x => x > offset)
            .ToList();
        if (nextWordIndexList.Count == 0)
        {
            return -1;
        }
        return nextWordIndexList.First();
    }

    /// <summary>
    /// 從指定位置開始往前尋找前一個單詞的起始位置
    /// </summary>
    /// <param name="offset">開始搜尋的位置</param>
    /// <returns>前一個單詞的起始位置，如果找不到則返回 -1</returns>
    public int IndexOfPrevWord(int offset)
    {
        var wordsIndexList = Chars.QueryWordsIndexList();
        var prevWordIndexList = wordsIndexList.Where(x => x < offset)
            .ToList();
        if (prevWordIndexList.Count == 0)
        {
            return -1;
        }
        return prevWordIndexList.Last();
    }
}