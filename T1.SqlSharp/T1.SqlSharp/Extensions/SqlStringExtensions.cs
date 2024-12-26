using System.Text.RegularExpressions;

namespace T1.SqlSharp.Extensions;

public static class SqlStringExtensions
{
    public static bool IsSameAs(this string text, string other)
    {
        return string.Equals(text, other, StringComparison.OrdinalIgnoreCase);
    }

    public static string NormalizeName(this string tableName)
    {
        tableName = tableName.Replace("[dbo].", "");
        tableName = Regex.Replace(tableName, @"\[(.*?)\]", "$1");
        return tableName;
    }
    
    public static bool IsNormalizeSameAs(this string text, string other)
    {
        return NormalizeName(text).IsSameAs(NormalizeName(other));
    }
}