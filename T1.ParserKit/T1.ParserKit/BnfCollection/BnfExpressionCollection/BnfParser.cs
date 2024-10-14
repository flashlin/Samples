namespace T1.ParserKit.BnfCollection.BnfExpressionCollection;

public class BnfParser
{
    private readonly List<MatchSpan> _tokens;
    private int _position;

    public BnfParser(string bnf)
    {
        var tokenizer = new BnfTokenizer();
        _tokens = tokenizer.ExtractMatches(bnf);
        _position = 0;
    }

    private MatchSpan CurrentToken => _tokens[_position];

    private void ConsumeToken() => _position++;

    private bool IsEof()
    {
        return _position >= _tokens.Count;
    }

    private bool Match(string value)
    {
        if (IsEof())
        {
            return false;
        }

        if (CurrentToken.Value == value)
        {
            ConsumeToken();
            return true;
        }

        return false;
    }

    private BnfRuleIdentifier ParseRuleIdentifier()
    {
        if (IsEof())
        {
            throw new Exception("Expected rule identifier.");
        }
        var identifier = new BnfRuleIdentifier
        {
            Name = CurrentToken.Value
        };
        ConsumeToken();
        return identifier;
    }

    private BnfIdentifier ParseIdentifier()
    {
        if (!IsEof())
        {
            var identifier = new BnfIdentifier
            {
                Name = CurrentToken.Value
            };
            ConsumeToken();
            return identifier;
        }

        throw new Exception("Expected identifier.");
    }

    private BnfString ParseString()
    {
        var bnfText = new BnfString
        {
            Text = CurrentToken.Value
        };
        ConsumeToken();
        return bnfText;
    }

    private BnfLiteral ParseLiteral()
    {
        if (!IsEof())
        {
            var literal = new BnfLiteral
            {
                Value = CurrentToken.Value
            };
            ConsumeToken();
            return literal;
        }

        throw new Exception("Expected literal.");
    }

    private IBnfExpression ParseExpression()
    {
        var left = ParseTerm();
        while (!IsEof())
        {
            var operatorToken = CurrentToken;

            if (operatorToken.Value == ";")
            {
                ConsumeToken();
                break;
            }
            
            if (Match(")"))
            {
                return left;
            }

            if (operatorToken.Value != "|")
            {
                if (left is BnfConcatExpression concat)
                {
                    concat.Items.Add(ParseExpression());
                    return concat;
                }

                return new BnfConcatExpression
                {
                    Items = [left, ParseExpression()],
                };
            }

            ConsumeToken();
            var right = ParseTerm();
            left = new BnfBinaryExpression
            {
                Left = left,
                Operator = operatorToken.Value,
                Right = right
            };
        }

        return left;
    }

    private IBnfExpression ParseTerm()
    {
        if (IsString(CurrentToken.Value))
        {
            return ParseString();
        }

        if (IsRuleIdentifier(CurrentToken.Value))
        {
            return ParseRuleIdentifier();
        }

        if (IsIdentifier(CurrentToken.Value))
        {
            return ParseIdentifier();
        }

        if (IsLiteral(CurrentToken.Value))
        {
            return ParseLiteral();
        }

        if (Match("("))
        {
            var innerExpression = ParseExpression();
            Match(")");
            return new BnfGroup
            {
                InnerExpression = innerExpression
            };
        }

        throw new Exception($"Unexpected term. {CurrentToken.Value}");
    }

    private bool IsIdentifier(string token) => !string.IsNullOrEmpty(token) && char.IsLetter(token[0]);
    private bool IsLiteral(string token) => !string.IsNullOrEmpty(token) && char.IsDigit(token[0]);

    private bool IsRuleIdentifier(string token) =>
        !string.IsNullOrEmpty(token) && token.StartsWith("<") && token.EndsWith(">");

    private bool IsString(string token) =>
        !string.IsNullOrEmpty(token) && token.StartsWith("\"") && token.EndsWith("\"");

    public BnfRule ParseBnfRule()
    {
        var rule = new BnfRule
        {
            RuleName = ParseIdentifier().Name
        };
        Match("::=");
        while (!IsEof())
        {
            rule.Expressions.Add(ParseExpression());
            if (!Match("|"))
            {
                break;
            }
        }

        return rule;
    }

    public IEnumerable<BnfRule> Parse()
    {
        while (!IsEof())
        {
            yield return ParseBnfRule();
        }
    }
}