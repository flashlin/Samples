namespace T1.SqlDom.Expressions;

public class InputStream
{
    public string Value { get; init; } = string.Empty;
    public int Position { get; set; }

    public void ExpectKeyword(string keyword)
    {
        if (!AcceptKeyword(keyword))
        {
            throw new Exception($"Expected keyword '{keyword}' at position {Position}");
        }
    }

    public void Expect(char c)
    {
        if (!Accept(c))
        {
            throw new Exception($"Expected character '{c}' at position {Position}");
        }
    }


    public bool AcceptAnyKeywordIgnoreCase(string[] list, out string outputString)
    {
        foreach (var searchString in list)
        {
            var len = searchString.Length;
            var currentString = SubStr(Position, len);
            if (string.Equals(currentString, searchString, StringComparison.OrdinalIgnoreCase))
            {
                Position += len;
                outputString = searchString;
                return true;
            }
        }

        outputString = string.Empty;
        return false;
    }

    public bool AcceptAnyKeyword(string[] list, out string outputString)
    {
        foreach (var searchString in list)
        {
            var len = searchString.Length;
            var currentString = Value.Substring(Position, len);
            if (currentString == searchString)
            {
                Position += len;
                outputString = searchString;
                return true;
            }
        }

        outputString = string.Empty;
        return false;
    }

    public bool AcceptKeyword(string keyword)
    {
        SkipSpaces();
        var subStr = Value.Substring(Position);
        if (subStr.StartsWith(keyword, StringComparison.OrdinalIgnoreCase))
        {
            Position += keyword.Length;
            return true;
        }

        return false;
    }

    public void SkipSpaces()
    {
        if (Position >= Value.Length)
        {
            return;
        }

        while (char.IsWhiteSpace(Value[Position]))
        {
            Position++;
        }
    }

    public bool Accept(Func<char, bool> check, out char output)
    {
        if (Position < Value.Length && check(Value[Position]))
        {
            output = Value[Position];
            Position += 1;
            return true;
        }

        output = char.MinValue;
        return false;
    }

    public bool Accept(char c)
    {
        if (Value[Position] == c)
        {
            Position++;
            return true;
        }

        return false;
    }

    public FetchResult AcceptUntil(char c)
    {
        var success = false;
        var result = string.Empty;
        while (Position < Value.Length && Value[Position] != c)
        {
            success = true;
            result += c;
        }

        if (!success)
        {
            return FetchResult.Empty;
        }

        return new FetchResult
        {
            Value = result,
            Success = true,
        };
    }

    public string SubStr(int start, int len)
    {
        var maxLen = Math.Min(Value.Length - start, len);
        return Value.Substring(start, maxLen);
    }

    public override string ToString()
    {
        return SubStr(Position, 20);
    }
}

public class FetchResult
{
    public static FetchResult Empty = new FetchResult();
    public string Value { get; set; } = string.Empty;
    public bool Success { get; set; }
}