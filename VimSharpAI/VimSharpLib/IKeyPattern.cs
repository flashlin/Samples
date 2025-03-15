namespace VimSharpLib
{
    public interface IKeyPattern
    {
        bool IsMatch(List<ConsoleKey> keyBuffer);
    }
} 
