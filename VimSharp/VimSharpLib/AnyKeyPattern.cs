namespace VimSharpLib;

public class AnyKeyPattern : IKeyPattern
{
    public bool IsMatch(List<ConsoleKeyInfo> keyBuffer)
    {
        return true;
    }
}