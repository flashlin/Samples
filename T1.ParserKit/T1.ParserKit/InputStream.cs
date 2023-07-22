using System.Text;

namespace T1.ParserKit;

public class InputStream
{
    public delegate Token MatchInputStreamFunc(InputStream stream, ReadOnlySpan<char> search);

    public delegate bool VerifySpanFunc(ReadOnlySpan<char> span);

    private readonly string _text;
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

        var remaining = Math.Min(_text.Length - _index - 1, n);
        return _text.AsSpan().Slice(_index + 1, remaining);
    }

    public ReadOnlySpan<char> Next(int n = 1)
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
        return _index + 1 >= _text.Length;
    }

    public ReadOnlySpan<char> GetSpan(int index, int count)
    {
        return _text.AsSpan().Slice(index, count);
    }

    public Token Read(VerifySpanFunc verify)
    {
        var buffer = new StringBuilder();
        var index = Index + 1;
        do
        {
            var ch = Peek();
            if (ch.IsEmpty())
            {
                break;
            }

            if (!verify(ch))
            {
                break;
            }
            
            buffer.Append(ch);
            Next();
        } while (!Eof());

        if (buffer.Length == 0)
        {
            return Token.Empty;
        }
        
        return new Token
        {
            Text = buffer.ToString(),
            Index = index
        };
    }

    public static Func<InputStream, Token> Func(MatchInputStreamFunc matchFunc, string search)
    {
        return (InputStream input) => matchFunc(input, search);
    }
}