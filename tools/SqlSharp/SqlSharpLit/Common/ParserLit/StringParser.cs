using T1.Standard.DesignPatterns;

namespace SqlSharpLit.Common.ParserLit;

public class StringParser
{
    private readonly string _text;
    private int _position;
    private Stack<int> _parsingContext = new();
    private TextSpan _previousWord = new();

    public StringParser(string text)
    {
        _text = text;
        _position = 0;
    }

    protected TextSpan ReadNumber()
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

    protected bool TryMatchKeyword(string keyword)
    {
        var peek = PeekKeyword();
        if (peek.Word != keyword.ToUpper()) return false;
        _previousWord = peek;
        _position = peek.Offset + peek.Length;
        return true;
    }
    
    protected bool TryMatch(string keyword)
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

    
    protected TextSpan PeekKeyword()
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

    protected void Match(string expected)
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

    protected TextSpan ReadUntil(Func<char, bool> predicate)
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
    
    
    protected TextSpan ReadIdentifier()
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

    protected TextSpan ReadQuotedIdentifier()
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


    protected bool IsEnd()
    {
        return _position >= _text.Length;
    }

    protected char PeekChar()
    {
        if (IsEnd()) return '\0';
        SkipWhitespace();
        return _text[_position];
    }

    protected char NextChar()
    {
        if (IsEnd()) return '\0';
        return _text[_position++];
    }

    protected void SkipWhitespace()
    {
        while (!IsEnd() && char.IsWhiteSpace(_text[_position]))
        {
            _position++;
        }
    }

    protected TextSpan PreviousWord()
    {
        return _previousWord;
    }

    protected bool IsWordChar(char c)
    {
        return char.IsLetterOrDigit(c) || c == '_' || c == '@' || c == '#' || c == '$';
    }
}

public interface ISqlExpression
{
}

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

public class ParseError : Exception
{
    public ParseError(string message) : base(message)
    {
    }
}

public class SqlParser : StringParser
{
    public SqlParser(string text) : base(text)
    {
    }

    public Either<CreateTableStatement, ParseError> ParseCreateTableStatement()
    {
        if (!(TryMatchKeyword("CREATE") && TryMatchKeyword("TABLE")))
        {
            return new Either<CreateTableStatement, ParseError>(
                new ParseError($"Expected CREATE TABLE, but got {PreviousWord().Word} {PeekKeyword().Word}"));
        }

        var tableName = ReadUntil(c => char.IsWhiteSpace(c) || c == '(');
        Match("(");

        var columns = new List<ColumnDefinition>();
        do
        {
            var item = ReadIdentifier();
            if (item.Length == 0)
            {
                return new Either<CreateTableStatement, ParseError>(
                    new ParseError($"Expected column name, but got {PeekKeyword().Word}"));
            }

            var column = new ColumnDefinition()
            {
                ColumnName = item.Word,
            };
            
            column.DataType = ReadIdentifier().Word;
            var dataLength1 = string.Empty;
            var dataLength2 = string.Empty;
            if (TryMatch("("))
            {
                dataLength1 = ReadNumber().Word;
                dataLength2 = string.Empty;
                if (PeekChar() == ',')
                {
                    NextChar();
                    dataLength2 = ReadNumber().Word;
                }
                Match(")");
            }

            if (!string.IsNullOrEmpty(dataLength1))
            {
                column.Size = int.Parse(dataLength1);
            }
            
            if (!string.IsNullOrEmpty(dataLength2))
            {
                column.Scale = int.Parse(dataLength2);
            }
            
            columns.Add(column);
            if (PeekChar() != ',')
            {
                break;
            }
            NextChar();
        } while (!IsEnd());

        return new Either<CreateTableStatement, ParseError>(new CreateTableStatement()
        {
            TableName = tableName.Word,
            Columns = columns
        });
    }


    
}