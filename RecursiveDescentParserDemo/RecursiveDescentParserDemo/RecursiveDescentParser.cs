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

    private SyntaxTreeNode ParseExpression()
    {
        SyntaxTreeNode left = ParseTerm();

        while (IsAddition() || IsSubtraction())
        {
            char op = GetOperator();
            SyntaxTreeNode right = ParseTerm();

            SyntaxTreeNode node = new SyntaxTreeNode { Token = op.ToString(), Left = left, Right = right };
            left = node;
        }

        return left;
    }

    private SyntaxTreeNode ParseTerm()
    {
        SyntaxTreeNode left = ParseFactor();

        while (IsMultiplication() || IsDivision())
        {
            char op = GetOperator();
            SyntaxTreeNode right = ParseFactor();

            SyntaxTreeNode node = new SyntaxTreeNode { Token = op.ToString(), Left = left, Right = right };
            left = node;
        }

        return left;
    }

    private SyntaxTreeNode ParseFactor()
    {
        if (IsNumber())
        {
            return new SyntaxTreeNode { Token = ConsumeNumber() };
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