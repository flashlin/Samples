namespace T1.SqlSharp.ParserLit;

public static class ArrayExtensions
{
    public static T GetValueOrDefault<T>(this List<T> tokens, int index, Func<T> defaultValue)
    {
        return index >= 0 && index < tokens.Count ? tokens[index] : defaultValue();
    }
}