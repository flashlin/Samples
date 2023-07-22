namespace T1.ParserKit;

public class SqlParser
{
    public IEnumerable<SqlExpr> Parse(string sql)
    {
        var tk = new SqlTokenizer();
        var tokens = tk.Tokenize(sql);
        var input = new TokenStream(tokens);
        var processList = new[]
        {
            ReadWhere,
        };
        while (!input.Eof())
        {
            var read = false;
            foreach (var process in processList)
            {
                var token = process(input);
                if (token == SqlExpr.Empty) continue;
                read = true;
                yield return token;
                break;
            }
            if (!read)
            {
                var remaining = input.Peek(10).Text;
                throw new NotSupportedException($"parse at \"{remaining}\"");
            }
        }
    }

    private SqlExpr ReadWhere(TokenStream input)
    {
        var token = input.Peek();
        if (!token.Text.Equals("where", StringComparison.OrdinalIgnoreCase))
        {
            return SqlExpr.Empty;
        }
        
        return new WhereExpr
        {
            Condition = ReadExpression(input)
        }; 
    }
    
    private SqlExpr ReadExpression(TokenStream input)
    {
        var left = ReadStarTerm(input);
        while (true)
        {
            var token = input.Peek();
            if (token.Text != "")
                break;

            var operatorToken = input.Next();

            var right = ReadStarTerm(input);

            left = new BinaryExpr(left, operatorToken, right);
        }

        return left;
    }
    
    private SqlExpr ReadStarTerm(TokenStream input)
    {
        return ReadTerm(input, "*");
    }
    
    private SqlExpr ReadTerm(TokenStream input, string term)
    {
        var left = ParseFactor(input);
        while (!input.Eof())
        {
            var token = input.Peek();
            if (token.Text != term)
                break;

            var operatorToken = input.Next();

            var right = ParseFactor(input);
            left = new BinaryExpr(left, operatorToken, right);
        }
        return left;
    }
    
    private SqlExpr ParseFactor(TokenStream input)
    {
        var token = input.Peek();
        if (token.Text != "(")
        {
            return ReadExpression(input);
        }
        var leftParam = input.Next();
        var inner = ReadExpression(input);
        var rightParam = input.Next();
        inner.LParam = leftParam;
        inner.RParam = rightParam;
        return inner;
    }
}