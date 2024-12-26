using System.Text.RegularExpressions;

namespace T1.SqlSharp.Extensions;

public static class SqlStringExtensions
{
    public static bool IsSameAs(this string text, string other)
    {
        return string.Equals(text, other, StringComparison.OrdinalIgnoreCase);
    }

    public static string NormalizeName(this string name)
    {
        name = name.Replace("[dbo].", "");
        name = Regex.Replace(name, @"\[(.*?)\]", "$1");
        if (name.StartsWith("N'", StringComparison.InvariantCultureIgnoreCase) && name.EndsWith("'"))
        {
            name = name.Substring(2, name.Length - 3);
        }
        return name;
    }
    
    public static bool IsNormalizeSameAs(this string text, string other)
    {
        return text.NormalizeName().IsSameAs(other.NormalizeName());
    }
}