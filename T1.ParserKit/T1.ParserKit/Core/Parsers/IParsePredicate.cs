namespace T1.ParserKit.Core.Parsers
{
    public interface IParsePredicate<TInput>
    {
        bool Execute(TInput input);
    }
}