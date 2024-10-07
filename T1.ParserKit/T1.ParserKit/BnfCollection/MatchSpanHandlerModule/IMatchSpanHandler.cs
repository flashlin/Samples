namespace T1.EfCore.Parsers.MatchSpanHandlerModule;

public interface IMatchSpanHandler
{
    MatchSpan Match(ReadOnlySpan<char> input, int index);
}