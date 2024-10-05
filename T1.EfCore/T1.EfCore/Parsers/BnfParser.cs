using System.Text;
using System.Text.RegularExpressions;

namespace T1.EfCore.Parsers;

public class BnfExpression
{
    public BnfExpression(string type, string value = null)
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
    private string input;
    private int position;

    public BnfParser(string input)
    {
        this.input = input;
        this.position = 0;
    }

    public BnfExpression Parse()
    {
        var root = new BnfExpression("Grammar");
        while (position < input.Length)
        {
            root.Children.Add(ParseRule());
        }

        return root;
    }

    private BnfExpression ParseRule()
    {
        SkipWhitespace();
        var rule = new BnfExpression("Rule");
        rule.Children.Add(ParseNonTerminal());
        Consume("::=");
        rule.Children.Add(ParseExpression());
        return rule;
    }

    private BnfExpression ParseNonTerminal()
    {
        return new BnfExpression("NonTerminal", ConsumeRegex(@"<[^>]+>"));
    }

    private BnfExpression ParseExpression()
    {
        var expression = ParseTerm();
        while (Peek("+"))
        {
            Consume("+");
            var right = ParseTerm();
            expression = new BnfExpression("Addition", "+")
            {
                Children = { expression, right }
            };
        }

        return expression;
    }

    private BnfExpression ParseTerm()
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
        else if (Peek("<"))
        {
            return ParseNonTerminal();
        }
        else if (Peek("\""))
        {
            return ParseTerminal();
        }
        else
        {
            throw new Exception("Unexpected token in factor");
        }
    }

    private BnfExpression ParseTerminal()
    {
        var terminal = new BnfExpression("Terminal");
        Consume("\"");
        var value = "";
        while (position < input.Length && input[position] != '"')
        {
            if (input[position] == '\\' && position + 1 < input.Length)
            {
                // Handle escaped characters
                value += input[position + 1];
                position += 2;
            }
            else
            {
                value += input[position];
                position++;
            }
        }
        if (position >= input.Length || input[position] != '"')
        {
            throw new Exception("Unterminated string literal");
        }
        Consume("\"");
        terminal.Value = value;
        return terminal;
    }

    private void Consume(string expected)
    {
        if (input.Substring(position, expected.Length) != expected)
            throw new Exception($"Expected '{expected}'");
        position += expected.Length;
        SkipWhitespace();
    }

    private string ConsumeRegex(string pattern)
    {
        var match = Regex.Match(input.Substring(position), $"^{pattern}");
        if (!match.Success)
            throw new Exception($"Expected pattern '{pattern}'");
        position += match.Length;
        SkipWhitespace();
        return match.Value;
    }

    private bool Peek(string s)
    {
        return position < input.Length && input.Substring(position).StartsWith(s);
    }

    private void SkipWhitespace()
    {
        while (position < input.Length && char.IsWhiteSpace(input[position]))
            position++;
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
}