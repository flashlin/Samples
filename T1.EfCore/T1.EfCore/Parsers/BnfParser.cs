using System.Text;
using System.Text.RegularExpressions;

namespace T1.EfCore.Parsers;

public class BnfExpression
{
    public BnfExpression(string type, string value = "")
    {
        Type = type;
        Value = value;
        Children = new List<BnfExpression>();
    }

    public string Type { get; set; }
    public string Value { get; set; }
    public List<BnfExpression> Children { get; set; }
}

public class BnfParser
{
    private List<MatchSpan> _tokens;
    private int _position = 0;

    public BnfParser(string input)
    {
        var tokenizer = new BnfTokenizer();
        _tokens = tokenizer.ExtractMatches(input);
    }

    public string GetExpressionTreeString(BnfExpression expr, int indent=0)
    {
        var text = new StringBuilder();
        text.AppendLine($"{new string(' ', indent)}{expr.Type}: {expr.Value}");
        foreach (var child in expr.Children)
        {
            var subText = GetExpressionTreeString(child, indent + 2);
            text.AppendLine(subText);
        }
        return text.ToString();
    }

    public BnfExpression Parse()
    {
        var root = new BnfExpression("Grammar");
        while (_position < _tokens.Count)
        {
            root.Children.Add(ParseRule());
        }
        return root;
    }

    private void Consume(string expected)
    {
        if (_tokens[_position].Value != expected)
            throw new Exception($"Expected '{expected}'");
        _position++;
    }

    private string ConsumeRegex(string pattern)
    {
        var match = Regex.Match(_tokens[_position].Value, $"^{pattern}");
        if (!match.Success)
            throw new Exception($"Expected pattern '{pattern}'");
        _position++;
        return match.Value;
    }

    private BnfExpression ParseExpression()
    {
        var expression = ParseMultiTerm();
        while (Peek("+"))
        {
            Consume("+");
            var right = ParseMultiTerm();
            expression = new BnfExpression("Addition", "+")
            {
                Children = { expression, right }
            };
        }
        return expression;
    }
    
    private BnfExpression ParseMultiTerm()
    {
        var term = ParseFactor();
        while (Peek("*"))
        {
            Consume("*");
            var right = ParseFactor();
            term = new BnfExpression("Multiplication", "*")
            {
                Children = { term, right }
            };
        }
        return term;
    }

    private BnfExpression ParseFactor()
    {
        if (Peek("("))
        {
            Consume("(");
            var expr = ParseExpression();
            Consume(")");
            return expr;
        }
        if (Peek("<"))
        {
            return ParseNonTerminal();
        }
        if (Peek("\""))
        {
            return ParseTerminal();
        }
        throw new Exception("Unexpected token in factor");
    }

    private BnfExpression ParseNonTerminal()
    {
        return new BnfExpression("NonTerminal", ConsumeRegex(@"<[^>]+>"));
    }

    private BnfExpression ParseRule()
    {
        var rule = new BnfExpression("Rule");
        rule.Children.Add(ParseNonTerminal());
        Consume("::=");
        rule.Children.Add(ParseExpression());
        return rule;
    }

    private BnfExpression ParseTerminal()
    {
        var terminal = new BnfExpression("Terminal");
        if(!_tokens[_position].Value.StartsWith("\""))
        {
            throw new Exception("Unterminated string literal");
        }
        if(!_tokens[_position].Value.EndsWith("\""))
        {
            throw new Exception("Unterminated string literal");
        }
        terminal.Value = _tokens[_position].Value;
        _position++;
        return terminal;
    }

    private bool HasMore()
    {
        return _position < _tokens.Count;
    }

    private bool Peek(string s)
    {
        return HasMore() && _tokens[_position].Value.StartsWith(s);
    }
}
