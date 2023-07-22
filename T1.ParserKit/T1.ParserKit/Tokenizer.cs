namespace T1.ParserKit;

public class Tokenizer
{
    public List<Token> Tokenize(string sql)
    {
        var input = new InputStream(sql);
        return ParseTokens(input).ToList();
    }

    private static IEnumerable<Token> ParseTokens(InputStream input)
    {
        var processList = new[]
        {
            ReadNumber,
            ReadIdentifier,
            ReadSpaces,
            ReadOperator,
            ReadSymbol
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
            }
            if (!read)
            {
                var remaining = input.Peek(10).ToString();
                throw new NotSupportedException($"tokenize at \"{remaining}\"");
            }
        }
    }

    private static Token ReadOperator(InputStream stream)
    {
        var ch = stream.Peek();
        var characters = "!=<>'\".@~`$%^&*-+/:|\\?";
        if (!characters.IsContains(ch)) return Token.Empty;

        return ReadAny(stream, 
            "===", "!==", "...", "&&=", "'''", "\"\"\"", "```",
            "!=", "==", ">=", "<=", "<>",
            "++", "--", "**", "//", "&&",
            "??", "?=", "+=", "-=", "/=", "*=", "~=", "^=", "&=", "%=", ":=",
            "@@");
    }
    
    private static Token ReadSymbol(InputStream stream)
    {
        var ch = stream.Peek();
        var characters = "{}()[],;\\`~!@#$%^&*-+_=|/:'\"<>?.";
        if (!characters.IsContains(ch)) return Token.Empty;
        var index = stream.Index + 1;
        return new Token
        {
            Text = stream.Next().ToString(),
            Index = index
        };
    }

    private static Token ReadAny(InputStream stream, params string[] spanList)
    {
        return ReadAny(stream, 
            spanList.Select(x => InputStream.Func(ReadSequenceEqual, x)).ToArray()
        );
    }

    private static Token ReadAny(InputStream stream, params Func<InputStream, Token>[] readFuncList)
    {
        foreach (var read in readFuncList)
        {
            var token = read(stream);
            if (token == Token.Empty) continue;
            return token;
        }
        return Token.Empty;
    }

    public delegate bool SpanEqualFn(ReadOnlySpan<char> ch);

    private static Token ReadSequenceEqual(InputStream stream, ReadOnlySpan<char> match)
    {
        var index = stream.Index + 1;
        var ch1 = stream.Peek(match.Length);
        if (ch1.SequenceEqual(match))
        {
            return new Token
            {
                Text = stream.Next(match.Length).ToString(),
                Index = index,
            };
        }
        return Token.Empty;
    }
    
    private static Token ReadSpaces(InputStream stream)
    {
        var ch = stream.Peek();
        if (!ch.IsWhiteSpace()) return Token.Empty;
        return stream.Read(ch => ch.IsWhiteSpace());
    }

    private static Token ReadNumber(InputStream stream)
    {
        var ch = stream.Peek();
        if (!ch.IsDigit()) return Token.Empty;
        return stream.Read(ch => ch.IsDigit());
    }

    private static Token ReadIdentifier(InputStream stream)
    {
        var ch = stream.Peek();
        if(!ch.IsUnderLetter()) return Token.Empty;
        var first = false;
        return stream.Read(ch =>
        {
            if (first)
            {
                first = false;
                return ch.IsUnderLetter();
            }
            return ch.IsUnderLetter() || ch.IsDigit();
        });
    }
}