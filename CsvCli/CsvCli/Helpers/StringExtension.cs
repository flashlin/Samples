using System.Text;

namespace CsvCli.Helpers;

public static class StringExtension
{
    static StringExtension()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
    }

    public static int GetBig5CharactersLength(this string text)
    {
        var big5 = Encoding.GetEncoding(950);
        return big5.GetBytes(text).Length;
    }

    public static string ToBig5FixLenString(this string text, int len, TextAlignType align = TextAlignType.Left)
    {
        if (string.IsNullOrEmpty(text))
        {
            return new string(' ', len);
        }

        var textLen = text.GetBig5CharactersLength();

        if (textLen > len)
        {
            var big5 = Encoding.GetEncoding(950);
            var textBuff = big5.GetBytes(text);
            var buff = new byte[len];
            Array.Copy(textBuff, buff, len);
            var str = big5.GetString(buff);
            return str;
        }


        var spaces = new string(' ', len - textLen);
        switch (align)
        {
            case TextAlignType.Left:
                return $"{text}{spaces}";
            default:
                return $"{spaces}{text}";
        }
    }

    public enum TextAlignType
    {
        Left,
        Right
    }
}