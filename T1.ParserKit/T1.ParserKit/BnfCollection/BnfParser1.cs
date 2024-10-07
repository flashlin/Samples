using System.Text;
using System.Text.RegularExpressions;

namespace T1.ParserKit.BnfCollection;

public interface IBnfExpressionVisitor
{
    void Visit(BnfExpression1 expression1);
}

public class LinqExpressionExpressionVisitor : IBnfExpressionVisitor
{
    private StringBuilder _result;

    public LinqExpressionExpressionVisitor()
    {
        _result = new StringBuilder();
    }

    public void Visit(BnfExpression1 expression1)
    {
        switch (expression1.Type)
        {
            case "From":
                _result.Append($"from {expression1.Value} ");
                break;
            case "Select":
                _result.Append($"select {expression1.Value} ");
                break;
            case "Join":
                _result.Append($"join {expression1.Value} ");
                break;
            case "Where":
                _result.Append($"where {expression1.Value} ");
                break;
            case "GroupBy":
                _result.Append($"group by {expression1.Value} ");
                break;
            case "OrderBy":
                _result.Append($"orderby {expression1.Value} ");
                break;
            default:
                _result.Append($"/* Unrecognized Type: {expression1.Type} */ ");
                break;
        }
    }

    public string GetResult()
    {
        return _result.ToString().Trim(); // 返回最终结果并去除多余空格
    }
}

public class BnfExpression1(string type, string value = "")
{
    public string Type { get; set; } = type;
    public string Value { get; set; } = value;
    public List<BnfExpression1> Children { get; set; } = new();
    
    public void Accept(IBnfExpressionVisitor expressionVisitor)
    {
        expressionVisitor.Visit(this);
        foreach (var child in Children)
        {
            child.Accept(expressionVisitor); // 遞迴訪問子節點
        }
    }
}

public class BnfParser1
{
    private List<MatchSpan> _tokens;
    private int _position = 0;

    public BnfParser1(string input)
    {
        var tokenizer = new BnfTokenizer();
        _tokens = tokenizer.ExtractMatches(input);
    }

    public string GetExpressionTreeString(BnfExpression1 expr, int indent=0)
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

    public BnfExpression1 Parse()
    {
        var root = new BnfExpression1("Grammar");
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
    
    private BnfExpression1 ParseExpression()
    {
        return ParseOrExpression();
    }
    
    private BnfExpression1 ParseOrExpression()
    {
        var expression = ParseAndExpression();
        while (Peek("|"))
        {
            Consume("|");
            var right = ParseAndExpression();
            expression = new BnfExpression1("Or", "|")
            {
                Children = { expression, right }
            };
        }
        return expression;
    }
    
    private BnfExpression1 ParseAndExpression()
    {
        var expression = ParseAdditionExpression();
        while (Peek("&"))
        {
            Consume("&");
            var right = ParseAdditionExpression();
            expression = new BnfExpression1("And", "&")
            {
                Children = { expression, right }
            };
        }
        return expression;
    }

    private BnfExpression1 ParseAdditionExpression()
    {
        var expression = ParseMultiplicationTerm();
        while (Peek("+"))
        {
            Consume("+");
            var right = ParseMultiplicationTerm();
            expression = new BnfExpression1("Addition", "+")
            {
                Children = { expression, right }
            };
        }
        return expression;
    }
    
    private BnfExpression1 ParseMultiplicationTerm()
    {
        var term = ParseFactor();
        while (Peek("*"))
        {
            Consume("*");
            var right = ParseFactor();
            term = new BnfExpression1("Multiplication", "*")
            {
                Children = { term, right }
            };
        }
        return term;
    }

    private BnfExpression1 ParseFactor()
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
            return ParseStringTerminal();
        }
        if (IsNumber())
        {
            return ParseNumberTerminal();
        }
        throw new Exception("Unexpected token in factor");
    }

    private BnfExpression1 ParseNonTerminal()
    {
        return new BnfExpression1("NonTerminal", ConsumeRegex(@"<[^>]+>"));
    }

    private BnfExpression1 ParseRule()
    {
        var rule = new BnfExpression1("Rule");
        rule.Children.Add(ParseNonTerminal());
        Consume("::=");
        rule.Children.Add(ParseExpression());
        return rule;
    }

    private BnfExpression1 ParseStringTerminal()
    {
        var terminal = new BnfExpression1("Terminal");
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
    
    private BnfExpression1 ParseNumberTerminal()
    {
        var terminal = new BnfExpression1("Terminal");
        var value = _tokens[_position].Value;
        if(!decimal.TryParse(value, out _))
        {
            throw new Exception("Unterminated number literal");
        }
        terminal.Value = value;
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

    private bool IsNumber()
    {
        var value = _tokens[_position].Value;
        return decimal.TryParse(value, out _);
    }
}
