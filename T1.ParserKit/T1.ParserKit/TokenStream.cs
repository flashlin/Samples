namespace T1.ParserKit;

public class TokenStream
{
    private readonly IEnumerator<Token> _tokens;
    private readonly List<Token> _buffer = new();
    private int _length = 0;
    private int _index = -1;

    public TokenStream(IEnumerable<Token> tokens)
    {
        _tokens = tokens.GetEnumerator();
    }

    public int Index => _index;

    public bool Eof()
    {
        if (_index + 1 < _length)
        {
            return false;
        }

        var token = Peek();
        return token == Token.Empty;
    }

    public Token Peek(int n = 1)
    {
        var index = _index;
        var token = Next(n);
        _index = index;
        return token;
    }
    
    public List<Token> PeekTokens(int n = 1)
    {
        var index = _index;
        var tokens = NextTokens(n);
        _index = index;
        return tokens;
    }

    public Token Next(int n = 1)
    {
        var tokens = NextTokens(n);
        return new Token
        {
            Text = string.Join("", tokens.Select(x => x.Text)),
            Index = tokens[0].Index
        };
    }

    public List<Token> NextTokens(int n = 1)
    {
        var count = 0;
        var span = new List<Token>();

        void FromBufferToSpan()
        {
            _index++;
            span.Add(_buffer[_index]);
            count++;
        }

        while (count < n)
        {
            if (_index + 1 < _length)
            {
                FromBufferToSpan();
            }
            else
            {
                if (!_tokens.MoveNext())
                {
                    break;
                }

                _buffer.Add(_tokens.Current);
                _length += 1;
                FromBufferToSpan();
            }
        }

        return span;
    }
}