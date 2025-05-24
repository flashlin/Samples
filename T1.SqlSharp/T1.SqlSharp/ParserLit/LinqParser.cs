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
        // parse where
        LinqWhereExpr whereExpr = null;
        if (TryParseWhere(out var where))
        {
            whereExpr = new LinqWhereExpr { Condition = where };
        }
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
            Where = whereExpr,
            Select = new LinqSelectAllExpr
            {
                AliasName = selectAliasResult.ResultValue
            }
        };
    }

    private bool TryParseWhere(out ILinqExpression where)
    {
        where = null;
        _text.SkipWhitespace();
        var pos = _text.Position;
        if (!_text.TryKeywordIgnoreCase("where", out _))
        {
            _text.Position = pos;
            return false;
        }
        _text.SkipWhitespace();
        where = ParseConditionExpr();
        return where != null;
    }

    private ILinqExpression ParseConditionExpr()
    {
        var left = ParseSingleCondition();
        if (left == null) return null;
        _text.SkipWhitespace();
        while (true)
        {
            LogicalOperator? logicalOp = null;
            if (_text.TryMatch("&&", out _))
            {
                logicalOp = LogicalOperator.And;
            }
            else if (_text.TryMatch("||", out _))
            {
                logicalOp = LogicalOperator.Or;
            }
            else
            {
                break;
            }
            _text.SkipWhitespace();
            var right = ParseSingleCondition();
            if (right == null) break;
            left = new LinqConditionExpression
            {
                Left = left,
                LogicalOperator = logicalOp,
                Right = right
            };
            _text.SkipWhitespace();
        }
        return left;
    }

    private ILinqExpression ParseSingleCondition()
    {
        var left = ParseLinqFieldExpr();
        if (left == null) return null;
        _text.SkipWhitespace();
        var op = ParseComparisonOperator();
        if (op == null) return null;
        _text.SkipWhitespace();
        var right = ParseLinqValue();
        if (right == null) return null;
        return new LinqConditionExpression
        {
            Left = left,
            ComparisonOperator = op.Value,
            Right = right
        };
    }

    private LinqFieldExpr ParseLinqFieldExpr()
    {
        _text.SkipWhitespace();
        var id1 = _text.ReadIdentifier();
        if (string.IsNullOrEmpty(id1.Word)) return null;
        var pos = _text.Position;
        _text.SkipWhitespace();
        if (_text.PeekChar() == '.')
        {
            _text.NextChar();
            var id2 = _text.ReadIdentifier();
            if (string.IsNullOrEmpty(id2.Word))
            {
                _text.Position = pos;
                return null;
            }
            return new LinqFieldExpr { TableOrAlias = id1.Word, FieldName = id2.Word };
        }
        return new LinqFieldExpr { TableOrAlias = null, FieldName = id1.Word };
    }

    private ComparisonOperator? ParseComparisonOperator()
    {
        if (_text.TryMatch("==", out _)) return ComparisonOperator.Equal;
        if (_text.TryMatch("!=", out _)) return ComparisonOperator.NotEqual;
        if (_text.TryMatch(">=", out _)) return ComparisonOperator.GreaterThanOrEqual;
        if (_text.TryMatch("<=", out _)) return ComparisonOperator.LessThanOrEqual;
        if (_text.TryMatch(">", out _)) return ComparisonOperator.GreaterThan;
        if (_text.TryMatch("<", out _)) return ComparisonOperator.LessThan;
        return null;
    }

    private LinqValue ParseLinqValue()
    {
        _text.SkipWhitespace();
        var num = _text.ReadInt();
        if (!string.IsNullOrEmpty(num.Word))
        {
            return new LinqValue { Value = num.Word };
        }
        var str = _text.ReadSqlQuotedString();
        if (!string.IsNullOrEmpty(str.Word))
        {
            return new LinqValue { Value = str.Word };
        }
        return null;
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