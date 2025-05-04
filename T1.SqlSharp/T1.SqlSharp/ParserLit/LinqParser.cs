using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.ParserLit;

public class LinqParser
{
    private readonly StringParser _text;
    public LinqParser(string text)
    {
        _text = new StringParser(text);
    }

    public ParseResult<LinqExpr> Parse()
    {
        var fromResult = Keywords("from")();
        if (fromResult.HasError) return fromResult.Error;
        var aliasResult = ParseIdentifier();
        if (aliasResult.HasError) return aliasResult.Error;
        var inResult = Keywords("in")();
        if (inResult.HasError) return inResult.Error;
        var sourceResult = ParseIdentifier();
        if (sourceResult.HasError) return sourceResult.Error;
        var selectResult = Keywords("select")();
        if (selectResult.HasError) return selectResult.Error;
        var selectAliasResult = ParseIdentifier();
        if (selectAliasResult.HasError) return selectAliasResult.Error;
        return new LinqExpr
        {
            From = new LinqFromExpr
            {
                Source = sourceResult.ResultValue,
                AliasName = aliasResult.ResultValue
            },
            Select = new LinqSelectAllExpr
            {
                AliasName = selectAliasResult.ResultValue
            }
        };
    }

    private Func<ParseResult<LinqToken>> Keywords(params string[] keywords)
    {
        return () => ParseKeywords(keywords);
    }

    private ParseResult<LinqToken> ParseKeywords(params string[] keywords)
    {
        if (TryKeywordsIgnoreCase(keywords, out var span))
        {
            return new LinqToken
            {
                Value = string.Join(" ", keywords),
                Span = span
            };
        }
        return NoneResult<LinqToken>();
    }

    private bool TryKeywordsIgnoreCase(string[] keywords, out TextSpan span)
    {
        return _text.TryKeywordsIgnoreCase(keywords, out span);
    }

    private ParseResult<string> ParseIdentifier()
    {
        _text.SkipWhitespace();
        var id = _text.ReadIdentifier();
        if (string.IsNullOrEmpty(id.Word))
        {
            return CreateParseError("Expected identifier");
        }
        return id.Word;
    }

    private ParseResult<T> NoneResult<T>()
    {
        return new ParseResult<T>(default(T));
    }

    private ParseError CreateParseError(string error)
    {
        return new ParseError(error)
        {
            Offset = _text.Position
        };
    }

    private class LinqToken
    {
        public string Value { get; set; }
        public TextSpan Span { get; set; }
    }
} 