namespace T1.SqlSharp.Extensions;

public static class SqlStringExtensions
{
    public static bool IsSameAs(this string text, string other)
    {
        return string.Equals(text, other, StringComparison.OrdinalIgnoreCase);
    }
}