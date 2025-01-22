namespace T1.SqlSharp.ParserLit;

public interface IParseResult
{
    bool HasResult { get; set; }
    ParseError Error { get; set; }
    bool HasError { get; set; }
    object? Object { get; }
    object ObjectValue { get; }
}