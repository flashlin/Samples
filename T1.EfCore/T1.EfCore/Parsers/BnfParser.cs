using System.Text;
using System.Text.RegularExpressions;

namespace T1.EfCore.Parsers;

public interface IBnfVisitor
{
    string Visit(BnfExpression expression);
}

public class TsqlBnfVisitor : IBnfVisitor
{
    public string Visit(BnfExpression expression)
    {
        // 假設 BnfExpression 的 Type 包含類型：SELECT, COLUMN, FROM, WHERE 等
        // 並且根據類型返回不同的 SQL 片段
        string result = expression.Type switch
        {
            "SELECT" => "SELECT " + VisitChildren(expression),
            "COLUMN" => expression.Value, // COLUMN 直接返回值
            "FROM" => "FROM " + VisitChildren(expression),
            "WHERE" => "WHERE " + VisitChildren(expression),
            _ => expression.Value // 默認返回表示式值
        };
        
        return result;
    }

    private string VisitChildren(BnfExpression expression)
    {
        // 遞迴遍歷 Children，拼接 SQL 片段
        return string.Join(" ", expression.Children.Select(child => Visit(child)));
    }
}

public class BnfExpression(string type, string value = "")
{
    public string Type { get; set; } = type;
    public string Value { get; set; } = value;
    public List<BnfExpression> Children { get; set; } = new();
    
    public string Accept(IBnfVisitor visitor)
    {
        return visitor.Visit(this);
    }
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
        return ParseOrExpression();
    }
    
    private BnfExpression ParseOrExpression()
    {
        var expression = ParseAndExpression();
        while (Peek("|"))
        {
            Consume("|");
            var right = ParseAndExpression();
            expression = new BnfExpression("Or", "|")
            {
                Children = { expression, right }
            };
        }
        return expression;
    }
    
    private BnfExpression ParseAndExpression()
    {
        var expression = ParseAdditionExpression();
        while (Peek("&"))
        {
            Consume("&");
            var right = ParseAdditionExpression();
            expression = new BnfExpression("And", "&")
            {
                Children = { expression, right }
            };
        }
        return expression;
    }

    private BnfExpression ParseAdditionExpression()
    {
        var expression = ParseMultiplicationTerm();
        while (Peek("+"))
        {
            Consume("+");
            var right = ParseMultiplicationTerm();
            expression = new BnfExpression("Addition", "+")
            {
                Children = { expression, right }
            };
        }
        return expression;
    }
    
    private BnfExpression ParseMultiplicationTerm()
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
            return ParseStringTerminal();
        }
        if (IsNumber())
        {
            return ParseNumberTerminal();
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

    private BnfExpression ParseStringTerminal()
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
    
    private BnfExpression ParseNumberTerminal()
    {
        var terminal = new BnfExpression("Terminal");
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
