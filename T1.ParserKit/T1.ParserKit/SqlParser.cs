namespace T1.ParserKit;

public class SqlParser
{
    public IEnumerable<SqlExpr> Parse(string sql)
    {
        var tk = new SqlTokenizer();
        var tokens = tk.Tokenize(sql);
        var input = new TokenStream(tokens);
        while (!input.Eof())
        {
            var expr = ReadStatement(input);
            if (expr == SqlExpr.Empty)
            {
                var remaining = input.Peek().Text;
                throw new NotSupportedException($"parse at \"{remaining}\"");
            }
            yield return expr;
        }
    }

    private SqlExpr ReadStatement(TokenStream input)
    {
        var processList = new[]
        {
            ReadWhere,
            ReadEqualStatement
        };
        foreach (var process in processList)
        {
            var token = process(input);
            if (token == SqlExpr.Empty) continue;
            return token;
        }
        return SqlExpr.Empty;
    }

    private SqlExpr ReadEqualStatement(TokenStream input)
    {
        var tokens = input.PeekTokens(2);
        if (tokens[1].Text == "=")
        {
            var left = ReadFactor(input);
            var op = input.Next();
            var right = ReadEqualStatement(input);
            return new OperatorExpr
            {
                Token = op,
                Left = left,
                Right = right
            };
        }
        return SqlExpr.Empty;
    }


    private SqlExpr ReadFactor(TokenStream input)
    {
        return new SqlExpr()
        {
            Token = input.Next()
        };
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
            Token = input.Next(),
            Condition = ReadExpression(input)
        };
    }

    private SqlExpr ReadExpression(TokenStream input)
    {
        var left = ReadStarTerm(input);
        while (!input.Eof())
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
            return new SqlExpr
            {
                Token = input.Next(),
            };
        }

        var leftParam = input.Next();
        var inner = ReadStatement(input);
        var rightParam = input.Next();
        inner.LParam = leftParam;
        inner.RParam = rightParam;
        return inner;
    }
}

public class OperatorExpr : SqlExpr
{
    public SqlExpr Left { get; set; } = SqlExpr.Empty;
    public SqlExpr Right { get; set; } = SqlExpr.Empty;
}