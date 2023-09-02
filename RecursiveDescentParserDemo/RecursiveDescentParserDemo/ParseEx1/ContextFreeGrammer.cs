namespace RecursiveDescentParserDemo.ParseEx1;

public class ContextFreeGrammer<T>
{
    private IEnumerableStream<T> _input;

    public ContextFreeGrammer()
    {
    }

    public ContextFreeGrammer(IEnumerableStream<T> input)
    {
        _input = input;
    }

    public void SetInput(IEnumerableStream<T> input)
    {
        _input = input;
    }

    public List<Token<T>> Many(Func<ContextFreeGrammer<T>, ContextFreeGrammer<T>> rule)
    {
        List<Token<T>> result = new();
        while (true)
        {
            var initialIndex = _input.GetPosition();
            var tokens = rule(new ContextFreeGrammer<T>(_input)).Tokens;
            if (tokens.Count == 0)
            {
                _input.Move(initialIndex);
                break;
            }
            result.AddRange(tokens);
        }
        return result;
    }

    public List<Token<T>> AtLeastOne(Func<ContextFreeGrammer<T>, ContextFreeGrammer<T>> rule)
    {
        var tokens = rule(new ContextFreeGrammer<T>(_input)).Tokens;
        if (tokens.Count == 0)
            throw new Exception("At least one token expected");
        return tokens;
    }

    public List<Token<T>> Option(Func<ContextFreeGrammer<T>, ContextFreeGrammer<T>> rule)
    {
        var initialIndex = _input.GetPosition();
        var tokens = rule(new ContextFreeGrammer<T>(_input)).Tokens;
        if (tokens.Count == 0)
        {
            _input.Move(initialIndex);
        }
        return tokens;
    }

    public List<Token<T>> Or(params Func<ContextFreeGrammer<T>, ContextFreeGrammer<T>>[] rules)
    {
        foreach (var rule in rules)
        {
            int initialIndex = _input.GetPosition();
            var tokens = rule(new ContextFreeGrammer<T>(_input)).Tokens;
            if (tokens.Count > 0)
            {
                return tokens;
            }
            _input.Move(initialIndex);
        }
        return new List<Token<T>>();
    }

    public List<Token<T>> Tokens { get; private set; } = new();

    public ContextFreeGrammer<T> Consume(IMatcher<T> matcher)
    {
        if (!_input.IsEof() && matcher.Match(_input.Current))
        {
            Tokens.Add(new Token<T> { Value = _input.Current });
            _input.MoveNext();
        }
        return this;
    }

    public ContextFreeGrammer<T> Consume(Token<T> token)
    {
        Tokens.Add(token);
        return this;
    }
}