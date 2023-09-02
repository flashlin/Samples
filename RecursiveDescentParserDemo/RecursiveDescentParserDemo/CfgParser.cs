public class CfgParser
{
    private string input;
    private int position;

    public CfgParser(string expression)
    {
        input = expression;
        position = 0;
    }

    // CFG描述: E -> T EPrime
    private SyntaxTreeNode E()
    {
        var left = T();
        return EPrime(left);
    }

    // CFG描述: EPrime -> ('+' T EPrime | '-' T EPrime | ε)
    private SyntaxTreeNode EPrime(SyntaxTreeNode left)
    {
        if (IsAddOrSubOperator())
        {
            var op = input[position - 1].ToString();
            var right = T();
            var newNode = new SyntaxTreeNode { Token = op, Left = left, Right = right };
            return EPrime(newNode);
        }
        return left; // ε
    }

    // CFG描述: T -> F TPrime
    private SyntaxTreeNode T()
    {
        var left = F();
        return TPrime(left);
    }

    // CFG描述: TPrime -> ('*' F TPrime | '/' F TPrime | ε)
    private SyntaxTreeNode TPrime(SyntaxTreeNode left)
    {
        if (IsMulOrDivOperator())
        {
            var op = input[position - 1].ToString();
            var right = F();
            var newNode = new SyntaxTreeNode { Token = op, Left = left, Right = right };
            return TPrime(newNode);
        }
        return left; // ε
    }

    // CFG描述: F -> '(' E ')' | number
    private SyntaxTreeNode F()
    {
        if (Match('('))
        {
            var node = E();
            if (Match(')'))
            {
                return node;
            }
            throw new ArgumentException("Mismatched parentheses.");
        }
        if (MatchNumber())
        {
            return new SyntaxTreeNode
            {
                Token = input[position - 1].ToString()
            };
        }
        throw new ArgumentException("Invalid expression.");
    }

    // 匹配字符的輔助函數
    private bool Match(char expected)
    {
        if (position < input.Length && input[position] == expected)
        {
            position++;
            return true;
        }
        return false;
    }

    // 匹配數字的輔助函數
    private bool MatchNumber()
    {
        int start = position;
        while (position < input.Length && char.IsDigit(input[position]))
        {
            position++;
        }
        return position > start;
    }

    // 判斷是否是加法或減法運算符的輔助函數
    private bool IsAddOrSubOperator()
    {
        return Match('+') || Match('-');
    }

    // 判斷是否是乘法或除法運算符的輔助函數
    private bool IsMulOrDivOperator()
    {
        return Match('*') || Match('/');
    }

    // 公共解析函數，返回整個語法樹
    public SyntaxTreeNode Parse()
    {
        return E();
    }
}