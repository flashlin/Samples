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
    public int NextWord(int offset)
    {
        if (offset < 0 || offset >= Width)
        {
            return -1;
        }

        // 跳過當前的空白字符
        while (offset < Width && (Chars[offset].Char == ' ' || Chars[offset].Char == '\0'))
        {
            offset++;
        }

        if (offset >= Width)
        {
            return -1;
        }

        // 判斷當前字符的類型
        var currentChar = Chars[offset].Char;
        bool isCurrentWord = char.IsLetterOrDigit(currentChar) || currentChar == '_';
        bool isCurrentPunctuation = char.IsPunctuation(currentChar);
        bool isCurrentChinese = currentChar > 127;

        // 尋找下一個不同類型的字符
        while (offset < Width)
        {
            var nextChar = Chars[offset].Char;
            if (nextChar == ' ' || nextChar == '\0')
            {
                // 找到空白字符，返回下一個非空白字符的位置
                while (offset < Width && (Chars[offset].Char == ' ' || Chars[offset].Char == '\0'))
                {
                    offset++;
                }
                return offset < Width ? offset : -1;
            }

            bool isNextWord = char.IsLetterOrDigit(nextChar) || nextChar == '_';
            bool isNextPunctuation = char.IsPunctuation(nextChar);
            bool isNextChinese = nextChar > 127;

            // 如果字符類型改變，返回當前位置
            if ((isCurrentWord && (isNextPunctuation || isNextChinese)) ||
                (isCurrentPunctuation && (isNextWord || isNextChinese)) ||
                (isCurrentChinese && (isNextWord || isNextPunctuation)))
            {
                return offset;
            }

            // 如果是中文字元，跳過下一個位置（因為中文字元佔用兩個位置）
            if (isNextChinese)
            {
                offset += 2;
            }
            else
            {
                offset++;
            }
        }

        return -1;
    }
}