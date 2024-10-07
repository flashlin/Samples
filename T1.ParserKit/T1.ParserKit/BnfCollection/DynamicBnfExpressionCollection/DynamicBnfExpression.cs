namespace T1.ParserKit.BnfCollection.DynamicBnfExpressionCollection;

public interface IDynamicBnfExpression
{
}

public class BnfIdentifier : IDynamicBnfExpression
{
    public string Name { get; set; } = string.Empty;
}

public class BnfLiteral : IDynamicBnfExpression
{
    public string Value { get; set; } = string.Empty; 
}

public class BnfBinaryExpression : IDynamicBnfExpression
{
    public required IDynamicBnfExpression Left { get; set; }
    public string Operator { get; set; } = string.Empty; 
    public required IDynamicBnfExpression Right { get; set; }
}

public class BnfGroup : IDynamicBnfExpression
{
    public required IDynamicBnfExpression InnerExpression { get; set; }
}

public class BnfRule : IDynamicBnfExpression
{
    public string RuleName { get; set; } = string.Empty;
    public List<IDynamicBnfExpression> Expressions { get; set; } = [];
}


public class DynamicBnfParser
{
    private readonly List<MatchSpan> _tokens;
    private int _position;

    public DynamicBnfParser(List<MatchSpan> tokens)
    {
        this._tokens = tokens;
        this._position = 0;
    }

    private MatchSpan? CurrentToken => 
        _position < _tokens.Count ? _tokens[_position] : null;

    private void AdvanceToken() => _position++;

    private bool Match(string value)
    {
        if (CurrentToken?.Value == value)
        {
            AdvanceToken();
            return true;
        }
        return false;
    }

    private BnfIdentifier ParseIdentifier()
    {
        if (CurrentToken != null)
        {
            var identifier = new BnfIdentifier
            {
                Name = CurrentToken.Value.Value
            };
            AdvanceToken();
            return identifier;
        }
        throw new Exception("Expected identifier.");
    }

    private BnfLiteral ParseLiteral()
    {
        if (CurrentToken != null)
        {
            var literal = new BnfLiteral
            {
                Value = CurrentToken.Value.Value
            };
            AdvanceToken();
            return literal;
        }
        throw new Exception("Expected literal.");
    }

    private IDynamicBnfExpression ParseExpression()
    {
        var left = ParseTerm();
        while (CurrentToken != null)
        {
            var operatorToken = CurrentToken.Value;
            AdvanceToken();
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

    private IDynamicBnfExpression ParseTerm()
    {
        if (CurrentToken != null)
        {
            if (IsIdentifier(CurrentToken?.Value))
            {
                return ParseIdentifier();
            }
            if (IsLiteral(CurrentToken?.Value))
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
        }
        throw new Exception("Unexpected term.");
    }

    private bool IsIdentifier(string? token) => !string.IsNullOrEmpty(token) && char.IsLetter(token[0]);
    private bool IsLiteral(string? token) => !string.IsNullOrEmpty(token) && char.IsDigit(token[0]);
    private bool IsOperator(string token) => token == "==" || token == "!=" || token == "<" || token == ">" || token == "<=" || token == ">=";

    // 解析 BNF 規則
    public BnfRule ParseBnfRule()
    {
        var rule = new BnfRule
        {
            RuleName = ParseIdentifier().Name
        };
        while (CurrentToken != null)
        {
            rule.Expressions.Add(ParseExpression());
            if (CurrentToken == null || !Match("|"))
            {
                break;
            }
        }
        return rule;
    }
}
