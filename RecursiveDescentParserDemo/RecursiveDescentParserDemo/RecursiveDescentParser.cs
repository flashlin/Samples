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
        if (IsNumber())
        {
            return new SyntaxTreeNode {Token = ConsumeNumber()};
        }
        else if (IsOpeningParenthesis())
        {
            ConsumeOpeningParenthesis();
            SyntaxTreeNode result = ParseE();
            ConsumeClosingParenthesis();
            return result;
        }
        else
        {
            throw new ArgumentException("Invalid expression.");
        }
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