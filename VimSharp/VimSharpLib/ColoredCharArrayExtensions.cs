using System.Text;

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
    
    public static string ToText(this ColoredChar[] chars)
    {
        var sb = new StringBuilder();
        foreach (var c in chars)
        {
            if (c.Char == '\0')
            {
                continue;
            }
            sb.Append(c.Char);
        }
        return sb.ToString();
    }

    public static ColoredChar[] ToColoredCharArray(this string text)
    {
        var consoleText = new ConsoleText();
        consoleText.SetText(0, text);
        return consoleText.Chars;
    }

    public static IEnumerable<int> QueryWordsIndexList(this ColoredChar[] chars)
    {
        var offset = 0;
        while (offset < chars.Length)
        {
            // 跳過空白字符
            while (offset < chars.Length && (chars[offset].Char == ' ' || chars[offset].Char == '\0'))
            {
                offset++;
            }

            if (offset >= chars.Length)
            {
                yield break;
            }

            // 記錄單詞起始位置
            yield return offset;
            
            // 判斷當前字符的類型
            var currentCharType = chars[offset].GetCharType();
            if (currentCharType is CharType.Word or CharType.Chinese)
            {
                offset = ContinueIdentifyWords(chars, offset) + 1;
                continue;
            }
            if (currentCharType is CharType.Punctuation)
            {
                offset = ContinuePunctuations(chars, offset) + 1;
                continue;
            }
            offset++;
        }
    }
    
    private static int ContinueIdentifyWords(ColoredChar[] chars, int offset)
    {
        if (offset >= chars.Length)
        {
            return -1;
        }
        while (offset < chars.Length)
        {
            var charType = chars[offset].GetCharType();
            if (charType is CharType.Punctuation or CharType.Space)
            {
                return offset-1;
            }
            offset++;
        }
        return offset;
    }
    
    private static int ContinuePunctuations(ColoredChar[] chars, int offset)
    {
        if (offset >= chars.Length)
        {
            return -1;
        }
        while (offset < chars.Length)
        {
            var charType = chars[offset].GetCharType();
            if (charType is CharType.Space or CharType.Word or CharType.Chinese)
            {
                return offset-1;
            }
            offset++;
        }
        return offset;
    }
    
    public static CharType GetCharType(this ColoredChar c)
    {
        if (c.Char == ' ')
        {
            return CharType.Space;
        }
        if (c.Char == '\0')
        {
            return CharType.None;
        }
        if (char.IsLetterOrDigit(c.Char) || c.Char == '_')
        {
            return CharType.Word;
        }
        if (char.IsPunctuation(c.Char))
        {
            return CharType.Punctuation;
        }
        if (c.Char > 127)
        {
            return CharType.Chinese;
        }
        return CharType.None;
    }
}

public enum CharType
{
    None,
    Space,
    Word,
    Punctuation,
    Chinese
}