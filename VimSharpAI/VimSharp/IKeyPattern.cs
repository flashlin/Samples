namespace VimSharp
{
    public interface IKeyPattern
    {
        bool IsMatch(List<ConsoleKey> keyBuffer);
    }
} 