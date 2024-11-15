namespace SqlSharpLit.Common.ParserLit;

public class StringParser
{
    private readonly string _text;
    private int _position;
    private TextSpan _previousWord = new();

    public StringParser(string text)
    {
        _text = text;
        _position = 0;
    }

    public string GetRemainingText()
    {
        if (IsEnd())
        {
            return string.Empty;
        }
        return _text.Substring(_position);
    }


    public bool IsEnd()
    {
        return _position >= _text.Length;
    }

    public bool IsWordChar(char c)
    {
        return char.IsLetterOrDigit(c) || c == '_' || c == '@' || c == '#' || c == '$';
    }

    public void Match(string expected)
    {
        SkipWhitespace();
        foreach (char c in expected)
        {
            if (IsEnd() || char.ToUpper(NextChar()) != char.ToUpper(c))
            {
                throw new Exception($"Expected '{expected}' at position {_position}, but found different content");
            }
        }
    }

    public char NextChar()
    {
        if (IsEnd()) return '\0';
        return _text[_position++];
    }

    public char PeekChar()
    {
        if (IsEnd()) return '\0';
        SkipWhitespace();
        return _text[_position];
    }

    public TextSpan PeekKeyword()
    {
        SkipWhitespace();
        var tempPosition = _position;
        var word = "";
        while (tempPosition < _text.Length && IsWordChar(_text[tempPosition]))
        {
            word += _text[tempPosition];
            tempPosition++;
        }

        return new TextSpan
        {
            Word = word,
            Offset = _position,
            Length = tempPosition - _position
        };
    }

    public TextSpan PreviousWord()
    {
        return _previousWord;
    }

    public TextSpan ReadIdentifier()
    {
        SkipWhitespace();
        var offset = _position;
        var ch = PeekChar();
        if (!char.IsLetter(ch) && ch != '_')
        {
            return new TextSpan()
            {
                Word = string.Empty,
                Offset = _position,
                Length = 0
            };
        }
        var identifier = "";
        while (!IsEnd())
        {
            var c = NextChar();
            if (!IsWordChar(c))
            {
                _position--;
                break;
            }
            identifier += c;
        }
        return new TextSpan()
        {
            Word = identifier,
            Offset = offset,
            Length = _position - offset
        };
    }

    public TextSpan ReadNumber()
    {
        SkipWhitespace();
        var offset = _position;
        var ch = PeekChar();
        if (!char.IsDigit(ch) && ch != '-')
        {
            return new TextSpan()
            {
                Word = string.Empty,
                Offset = _position,
                Length = 0
            };
        }

        var word = "";
        while (!IsEnd())
        {
            ch = NextChar();
            if (!char.IsDigit(ch))
            {
                _position--;
                break;
            }
            word += ch;
        }

        return new TextSpan()
        {
            Word = word,
            Offset = offset,
            Length = _position - offset
        };
    }

    public TextSpan ReadQuotedIdentifier()
    {
        var quoteChar = PeekChar();
        if (quoteChar != '"' && quoteChar != '[' && quoteChar != '`')
        {
            return new TextSpan()
            {
                Word = string.Empty,
                Offset = _position,
                Length = 0
            };
        }

        var offset = _position;
        var closeChar = quoteChar == '[' ? ']' : quoteChar;
        var identifier = quoteChar.ToString();
        NextChar();
        while (!IsEnd())
        {
            var c = NextChar();
            identifier += c;
            if (c == closeChar)
            {
                break;
            }
        }

        return new TextSpan()
        {
            Word = identifier,
            Offset = offset,
            Length = _position - offset
        };
    }

    public TextSpan ReadSymbol()
    {
        SkipWhitespace();
        var offset = _position;
        var ch = PeekChar();
        if (char.IsLetter(ch) && ch != '_')
        {
            return new TextSpan()
            {
                Word = string.Empty,
                Offset = _position,
                Length = 0
            };
        }
        var symbol = "";
        while (!IsEnd())
        {
            var c = NextChar();
            if (IsWordChar(c))
            {
                _position--;
                break;
            }
            symbol += c;
        }
        return new TextSpan()
        {
            Word = symbol,
            Offset = offset,
            Length = _position - offset
        };
    }

    public TextSpan ReadUntil(Func<char, bool> predicate)
    {
        var offset = _position;
        var result = "";
        while (!IsEnd() && !predicate(PeekChar()))
        {
            result += NextChar();
        }

        return new TextSpan()
        {
            Word = result,
            Offset = offset,
            Length = _position - offset
        };
    }

    public void SkipWhitespace()
    {
        while (!IsEnd() && char.IsWhiteSpace(_text[_position]))
        {
            _position++;
        }
    }

    public bool Try(Func<TextSpan> readFunc, out TextSpan textSpan)
    {
        textSpan = readFunc();
        if (textSpan.Length == 0)
        {
            return false;
        }
        return true;
    }

    public bool TryMatch(string keyword)
    {
        SkipWhitespace();
        var tempPosition = _position;
        var word = "";
        while (word.Length < keyword.Length)
        {
            word += _text[tempPosition];
            tempPosition++;
        }
        if (word != keyword)
        {
            return false;
        }

        _previousWord = new TextSpan
        {
            Word = keyword,
            Offset = _position,
            Length = keyword.Length
        };
        _position = tempPosition;
        return true;
    }

    public bool TryMatchKeyword(string keyword)
    {
        var peek = PeekKeyword();
        if (peek.Word != keyword.ToUpper()) return false;
        _previousWord = peek;
        _position = peek.Offset + peek.Length;
        return true;
    }
}

public interface ISqlExpression
{}

public class ColumnDefinition : ISqlExpression
{
    public string ColumnName { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public bool IsPrimaryKey { get; set; }
    public bool IsAutoIncrement { get; set; }
    public int Size { get; set; }
    public int Scale { get; set; }
}

public class CreateTableStatement : ISqlExpression
{
    public string TableName { get; set; } = string.Empty;
    public List<ColumnDefinition> Columns { get; set; } = [];
}

public class SelectColumn : ISelectColumnExpression
{
    public string ColumnName { get; set; } = string.Empty;
}

public interface ISelectColumnExpression
{}


public class SelectFrom : ISelectFromExpression
{
    public string FromTableName { get; set; } = string.Empty;
}

public interface ISelectFromExpression
{}

public class SqlWhereExpression : ISqlWhereExpression 
{
    public ISqlExpression Left { get; set; }
    public string Operation { get; set; } = string.Empty;
    public ISqlExpression Right { get; set; }
}

public class SqlFieldExpression : ISqlExpression
{
    public string FieldName { get; set; } = string.Empty;
}

public class SqlIntValueExpression : ISqlExpression
{
    public int Value { get; set; }
}

public interface ISqlWhereExpression
{}

public class SelectStatement : ISqlExpression
{
    public List<ISelectColumnExpression> Columns { get; set; } = [];
    public ISelectFromExpression From { get; set; } = new SelectFrom();
    public ISqlWhereExpression Where { get; set; }
}
