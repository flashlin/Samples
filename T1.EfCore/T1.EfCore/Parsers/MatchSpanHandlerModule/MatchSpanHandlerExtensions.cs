namespace T1.EfCore.Parsers.MatchSpanHandlerModule;

public static class MatchSpanHandlerExtensions
{
    public static MatchSpanPlusHandler Plus(this IMatchSpanHandler handler)
    {
        return new MatchSpanPlusHandler(handler);
    }

    public static MatchSpanConcatHandler Concat(this IMatchSpanHandler handler, IMatchSpanHandler nextHandler)
    {
        return new MatchSpanConcatHandler(handler, nextHandler);
    }
}