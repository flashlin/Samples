namespace GitCli.Models.ConsoleMixedReality;

public static class StringExtension
{
    public static string SubStr(this string str, int offset)
    {
        if (string.IsNullOrEmpty(str))
        {
            return string.Empty;
        }
        if (offset < 0)
        {
            return string.Empty;
        }
        if (offset >= str.Length)
        {
            return string.Empty;
        }
        return str.Substring(offset);
    }
}