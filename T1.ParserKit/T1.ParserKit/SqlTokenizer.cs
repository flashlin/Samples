namespace T1.ParserKit;

public class SqlTokenizer
{
    public IEnumerable<Token> Tokenize(string sql)
    {
        var tk = new Tokenizer();
        var input = new TokenStream(tk.Tokenize(sql));
        
        var processList = new[]
        {
            SkipSpaces,
            ReadString,
            ReadToken
        };
        while (!input.Eof())
        {
            var read = false;
            foreach (var process in processList)
            {
                var token = process(input);
                if (token == Token.Empty) continue;
                read = true;
                yield return token;
                break;
            }
            if (!read)
            {
                var remaining = input.Peek(10).Text;
                throw new NotSupportedException($"tokenize at \"{remaining}\"");
            }
        }
    }

    private Token SkipSpaces(TokenStream input)
    {
        var token = input.Peek();
        if (token.Text.Trim() == string.Empty)
        {
            input.Next();
            return Token.Empty;
        }
        return Token.Empty;
    }

    private Token ReadString(TokenStream input)
    {
        var token = input.Peek();
        if (token.Text != "'")
        {
            return Token.Empty;
        }

        var buff = new List<Token>();
        buff.Add(input.Next());
        while (!input.Eof())
        {
            token = input.Peek();
            if (token == Token.Empty)
            {
                break;
            }
            if (token.Text == "'")
            {
                buff.Add(input.Next());
                break;
            }
            buff.Add(input.Next());
        }

        return new Token
        {
            Text = string.Join("", buff.Select(x => x.Text)),
            Index = buff[0].Index
        };
    }
    
    private Token ReadToken(TokenStream input)
    {
        return input.Next();
    }
}