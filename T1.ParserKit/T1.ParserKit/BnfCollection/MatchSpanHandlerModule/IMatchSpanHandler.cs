namespace T1.ParserKit.BnfCollection.MatchSpanHandlerModule;

public interface IMatchSpanHandler
{
    MatchSpan Match(ReadOnlySpan<char> input, int index);
}