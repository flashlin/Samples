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
        return ParseExpression();
    }

    // CFG描述: E -> T E'
    private SyntaxTreeNode ParseExpression()
    {
        SyntaxTreeNode left = ParseTerm();
        return ParseExpressionPrime(left);
    }

    // CFG描述: E' -> ('+' | '-') T E' | ε
    private SyntaxTreeNode ParseExpressionPrime(SyntaxTreeNode left)
    {
        if (IsAddition() || IsSubtraction())
        {
            char op = GetOperator();
            SyntaxTreeNode right = ParseTerm();
            SyntaxTreeNode node = new SyntaxTreeNode {Token = op.ToString(), Left = left, Right = right};
            return ParseExpressionPrime(node);
        }
        return left; // ε
    }

    // CFG描述: T -> F T'
    private SyntaxTreeNode ParseTerm()
    {
        SyntaxTreeNode left = ParseFactor();
        return ParseTermPrime(left);
    }
    
    // CFG描述: T' -> ('*' | '/') F T' | ε
    private SyntaxTreeNode ParseTermPrime(SyntaxTreeNode left)
    {
        if (IsMultiplication() || IsDivision())
        {
            char op = GetOperator();
            SyntaxTreeNode right = ParseFactor();
            SyntaxTreeNode node = new SyntaxTreeNode {Token = op.ToString(), Left = left, Right = right};
            return ParseTermPrime(node);
        }
        return left;
    }

    // CFG描述: F -> '(' E ')' | number
    private SyntaxTreeNode ParseFactor()
    {
        if (IsNumber())
        {
            return new SyntaxTreeNode {Token = ConsumeNumber()};
        }
        else if (IsOpeningParenthesis())
        {
            ConsumeOpeningParenthesis();
            SyntaxTreeNode result = ParseExpression();
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