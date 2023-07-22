namespace T1.ParserKit;

public static class SpanExtension
{
    public static bool IsContains(this string text, ReadOnlySpan<char> span)
    {
        return text.IndexOf(span[0]) != -1;
    }
    
    public static bool IsEmpty(this ReadOnlySpan<char> ch)
    {
        return ch.SequenceEqual(ReadOnlySpan<char>.Empty);
    }

    public static bool IsUnderLetter(this ReadOnlySpan<char> ch)
    {
        return ch.SequenceEqual("_".AsSpan()) || char.IsLetter(ch[0]);
    }

    public static bool IsDigit(this ReadOnlySpan<char> ch)
    {
        return char.IsDigit(ch[0]);
    }

    public static bool IsWhiteSpace(this ReadOnlySpan<char> ch)
    {
        return char.IsWhiteSpace(ch[0]);
    }

    public static bool SequenceEqual(this ReadOnlySpan<char> ch2, string s)
    {
        return ch2.SequenceEqual(s.AsSpan());
    }
}