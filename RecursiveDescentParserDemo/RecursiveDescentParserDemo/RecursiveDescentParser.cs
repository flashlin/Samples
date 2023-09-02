public class RecursiveDescentParser
{
    private string input;
    private int position;

    public RecursiveDescentParser(string expression)
    {
        input = expression;
        position = 0;
    }

    public SyntaxTreeNode Parse()
    {
        return ParseE();
    }

    // CFG描述: E -> T E'
    private SyntaxTreeNode ParseE()
    {
        SyntaxTreeNode left = ParseT();
        return ParseEPrime(left);
    }

    // CFG描述: E' -> ('+' | '-') T E' | ε
    private SyntaxTreeNode ParseEPrime(SyntaxTreeNode left)
    {
        if (IsAddition() || IsSubtraction())
        {
            char op = GetOperator();
            SyntaxTreeNode right = ParseT();
            SyntaxTreeNode node = new SyntaxTreeNode {Token = op.ToString(), Left = left, Right = right};
            return ParseEPrime(node);
        }

        return left; // ε
    }

    // CFG描述: T -> F T'
    private SyntaxTreeNode ParseT()
    {
        SyntaxTreeNode left = ParseF();
        return ParseTPrime(left);
    }

    // CFG描述: T' -> ('*' | '/') F T' | ε
    private SyntaxTreeNode ParseTPrime(SyntaxTreeNode left)
    {
        if (IsMultiplication() || IsDivision())
        {
            char op = GetOperator();
            SyntaxTreeNode right = ParseF();
            SyntaxTreeNode node = new SyntaxTreeNode {Token = op.ToString(), Left = left, Right = right};
            return ParseTPrime(node);
        }

        return left;
    }

    // CFG描述: F -> '(' E ')' | number
    private SyntaxTreeNode ParseF()
    {
        if (IsOpeningParenthesis())
        {
            ConsumeOpeningParenthesis();
            SyntaxTreeNode result = ParseE();
            ConsumeClosingParenthesis();
            return result;
        }

        if (IsNumber())
        {
            return new SyntaxTreeNode {Token = ConsumeNumber()};
        }

        throw new ArgumentException("Invalid expression.");
    }

    private bool IsAddition()
    {
        return Peek() == '+';
    }

    private bool IsSubtraction()
    {
        return Peek() == '-';
    }

    private bool IsMultiplication()
    {
        return Peek() == '*';
    }

    private bool IsDivision()
    {
        return Peek() == '/';
    }

    private bool IsNumber()
    {
        return char.IsDigit(Peek());
    }

    private bool IsOpeningParenthesis()
    {
        return Peek() == '(';
    }

    private char GetOperator()
    {
        char op = Peek();
        Consume();
        return op;
    }

    private string ConsumeNumber()
    {
        int start = position;
        while (position < input.Length && (char.IsDigit(input[position]) || input[position] == '.'))
        {
            position++;
        }

        return input.Substring(start, position - start);
    }

    private void ConsumeOpeningParenthesis()
    {
        if (IsOpeningParenthesis())
        {
            Consume();
        }
        else
        {
            throw new ArgumentException("Expected opening parenthesis.");
        }
    }

    private void ConsumeClosingParenthesis()
    {
        if (Peek() == ')')
        {
            Consume();
        }
        else
        {
            throw new ArgumentException("Expected closing parenthesis.");
        }
    }

    private char Peek()
    {
        if (position < input.Length)
        {
            return input[position];
        }

        return '\0'; // 表示字符串結束
    }

    private void Consume()
    {
        if (position < input.Length)
        {
            position++;
        }
    }
}


public enum LinqRules
{
    Identifier,
    SelectExpression,
    aliaName,
    Space
}

public static class Parser
{
    public static object SKIPPED;

    public static object Token<TEnum>(TEnum ruleName, string pattern, object? skipped = default)
        where TEnum : struct
    {
        return null;
    }

    public static void Perform(object[] allTokens)
    {
        throw new NotImplementedException();
    }
}

public class EmbeddedActionsParser
{
    protected EmbeddedActionsParser(object[] allTokens)
    {
        throw new NotImplementedException();
    }

    public object Rule<TEnum>(TEnum ruleName, Action implementation)
        where TEnum : struct
    {
        return null;
    }

    public object Rule<TEnum>(TEnum ruleName, Func<object> implementation)
        where TEnum : struct
    {
        return null;
    }

    public object Consume(object token)
    {
        return null;
    }

    public object Subrule(object rule)
    {
        return null;
    }
    
    private void Scan(string text)
    {
        throw new NotImplementedException();
    }
}

public class LinqParser : EmbeddedActionsParser
{
    static object White = Parser.Token(LinqRules.Space, "", Parser.SKIPPED);
    static object Identifier = Parser.Token(LinqRules.Identifier, "[a-zA-Z_]\\w*");

    private static object[] AllTokens =
    {
        White,
        Identifier,
    };

    private object SelectExpression;
    private object aliaName;

    public LinqParser()
        : base(AllTokens)
    {
        SelectExpression = this.Rule(LinqRules.SelectExpression, () =>
        {
            this.Consume(Identifier);
            var aliasName = this.Subrule(aliaName);
        });
        aliaName = this.Rule(LinqRules.aliaName, () => { return this.Consume(Identifier); });
    }

    public void Parse(string text)
    {
        //this.Scan(text);
        //this.SelectExpression();
    }
}


public class Token
{
    public string Value { get; set; }
}

public class ContextFreeGrammer
{
    private string input;
    private int currentIndex = 0;

    public ContextFreeGrammer(string input)
    {
        this.input = input;
    }

    public List<Token> many(Func<ContextFreeGrammer, ContextFreeGrammer> rule)
    {
        List<Token> result = new List<Token>();
        while (true)
        {
            int initialIndex = currentIndex;
            var tokens = rule(new ContextFreeGrammer(input)).Tokens;
            if (tokens.Count == 0)
            {
                currentIndex = initialIndex;
                break;
            }
            result.AddRange(tokens);
        }
        return result;
    }

    public List<Token> at_least_one(Func<ContextFreeGrammer, ContextFreeGrammer> rule)
    {
        var tokens = rule(new ContextFreeGrammer(input)).Tokens;
        if (tokens.Count == 0)
            throw new Exception("At least one token expected");
        return tokens;
    }

    public List<Token> option(Func<ContextFreeGrammer, ContextFreeGrammer> rule)
    {
        int initialIndex = currentIndex;
        var tokens = rule(new ContextFreeGrammer(input)).Tokens;
        if (tokens.Count == 0)
        {
            currentIndex = initialIndex;
        }
        return tokens;
    }

    public List<Token> or(params Func<ContextFreeGrammer, ContextFreeGrammer>[] rules)
    {
        foreach (var rule in rules)
        {
            int initialIndex = currentIndex;
            var tokens = rule(new ContextFreeGrammer(input)).Tokens;
            if (tokens.Count > 0)
            {
                return tokens;
            }
            currentIndex = initialIndex;
        }
        return new List<Token>();
    }

    public List<Token> Tokens { get; private set; } = new List<Token>();

    public ContextFreeGrammer consume(string value)
    {
        if (currentIndex < input.Length && input.Substring(currentIndex).StartsWith(value))
        {
            Tokens.Add(new Token { Value = value });
            currentIndex += value.Length;
        }
        return this;
    }

    public ContextFreeGrammer consume(Token token)
    {
        Tokens.Add(token);
        return this;
    }
}

public class Sample
{
    public void Run()
    {
        // 示例用法
        var input = "1ababab";
        var cfg = new ContextFreeGrammer(input);
        var tokens = cfg.or(
            c => c.consume("a").consume("b"),
            c => c.consume("b").consume("a")
        );

        Console.WriteLine(tokens.Count);
        foreach (var token in tokens)
        {
            Console.WriteLine(token.Value);
        }
    }
}
