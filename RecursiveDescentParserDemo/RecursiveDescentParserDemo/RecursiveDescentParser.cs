using RecursiveDescentParserDemo.ParseEx1;

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

public class TextEnumerableStream : IEnumerableStream<string>
{
    private readonly string _text;
    private int _position = 0;

    public TextEnumerableStream(string text)
    {
        _text = text;
    }

    public string Current
    {
        get
        {
            if (IsEof())
            {
                return string.Empty;
            }
            return _text.Substring(_position, 1);
        }
    }

    public void Move(int position)
    {
        _position = position;
    }

    public int GetPosition()
    {
        return _position;
    }

    public bool IsEof()
    {
        if (_position < _text.Length)
        {
            return false;
        }
        return true;
    }

    public bool MoveNext()
    {
        if (IsEof())
        {
            return false;
        }

        _position++;
        return true;
    }
}