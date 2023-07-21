using System.Text;

namespace T1.ParserKit;

public class InputStream
{
    readonly string _text;
    private int _index;

    public InputStream(string text)
    {
        _text = text;
        _index = -1;
        //var isEqual = span.SequenceEqual(targetSpan);
    }

    public int Index => _index;

    public ReadOnlySpan<char> Peek(int n = 1)
    {
        if (_index + 1 >= _text.Length)
        {
            return ReadOnlySpan<char>.Empty;
        }

        var remaining = Math.Min(_text.Length - _index + 1, n);
        return _text.AsSpan().Slice(_index + 1, remaining);
    }

    public ReadOnlySpan<char> Read(int n = 1)
    {
        if (_index + 1 >= _text.Length)
        {
            return ReadOnlySpan<char>.Empty;
        }

        var remaining = Math.Min(_text.Length - _index + 1, n);
        var span = _text.AsSpan().Slice(_index + 1, remaining);
        _index += remaining;
        return span;
    }

    public bool Eof()
    {
        return _index + 1 < _text.Length;
    }

    public ReadOnlySpan<char> GetSpan(int index, int count)
    {
        return _text.AsSpan().Slice(index, count);
    }
}

public class Token
{
    public static readonly Token Empty = new Token
    {
        Text = string.Empty,
        Index = -1
    };
    public string Text { get; set; }
    public int Index { get; set; }
}

public class SqlTokenizer
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
            TryReadNumber
        };
        while (!input.Eof())
        {
            foreach (var process in processList)
            {
                var (success, token) = process(input);
                if (!success) continue;
                yield return token;
                break;
            }
        }
    }

    private static (bool success, Token token) TryReadNumber(InputStream stream)
    {
        var ch = stream.Peek();
        if (char.IsDigit(ch[0]))
        {
            var number = new StringBuilder();
            var index = stream.Index + 1;
            do
            {
                var ch1 = stream.Peek();
                if (ch1.SequenceEqual(ReadOnlySpan<char>.Empty))
                {
                    break;
                }

                if (!char.IsDigit(ch1[0]))
                {
                    break;
                }

                number.Append(ch1);
                stream.Read();
            } while (!stream.Eof());

            {
                return (true, new Token
                {
                    Text = number.ToString(),
                    Index = index
                });
            }
        }

        return (false, Token.Empty);
    }
}