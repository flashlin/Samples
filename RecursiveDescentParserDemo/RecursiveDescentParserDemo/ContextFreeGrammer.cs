public class ContextFreeGrammer
{
    private string input;
    private int currentIndex = 0;

    public ContextFreeGrammer(string input)
    {
        this.input = input;
    }

    public List<Token> many(Func<ContextFreeGrammer, ContextFreeGrammer> rule)
    {
        List<Token> result = new List<Token>();
        while (true)
        {
            int initialIndex = currentIndex;
            var tokens = rule(new ContextFreeGrammer(input)).Tokens;
            if (tokens.Count == 0)
            {
                currentIndex = initialIndex;
                break;
            }
            result.AddRange(tokens);
        }
        return result;
    }

    public List<Token> at_least_one(Func<ContextFreeGrammer, ContextFreeGrammer> rule)
    {
        var tokens = rule(new ContextFreeGrammer(input)).Tokens;
        if (tokens.Count == 0)
            throw new Exception("At least one token expected");
        return tokens;
    }

    public List<Token> option(Func<ContextFreeGrammer, ContextFreeGrammer> rule)
    {
        int initialIndex = currentIndex;
        var tokens = rule(new ContextFreeGrammer(input)).Tokens;
        if (tokens.Count == 0)
        {
            currentIndex = initialIndex;
        }
        return tokens;
    }

    public List<Token> or(params Func<ContextFreeGrammer, ContextFreeGrammer>[] rules)
    {
        foreach (var rule in rules)
        {
            int initialIndex = currentIndex;
            var tokens = rule(new ContextFreeGrammer(input)).Tokens;
            if (tokens.Count > 0)
            {
                return tokens;
            }
            currentIndex = initialIndex;
        }
        return new List<Token>();
    }

    public List<Token> Tokens { get; private set; } = new List<Token>();

    public ContextFreeGrammer consume(string value)
    {
        if (currentIndex < input.Length && input.Substring(currentIndex).StartsWith(value))
        {
            Tokens.Add(new Token { Value = value });
            currentIndex += value.Length;
        }
        return this;
    }

    public ContextFreeGrammer consume(Token token)
    {
        Tokens.Add(token);
        return this;
    }
}