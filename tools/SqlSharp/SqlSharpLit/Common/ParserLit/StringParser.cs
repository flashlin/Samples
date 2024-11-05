using T1.Standard.DesignPatterns;

namespace SqlSharpLit.Common.ParserLit;

public class StringParser
{
    private string _text;
    private int _position;
    private Stack<int> _parsingContext = new();
    private string _previousWord = string.Empty;

    public StringParser(string text)
    {
        _text = text;
        _position = 0;
    }

    protected bool TryMatchKeyword(string keyword)
    {
        var peek = PeekKeyword();
        if (peek.Word != keyword.ToUpper()) return false;
        _previousWord = peek.Word;
        _position = peek.Offset + peek.Length;
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

    protected string ReadUntil(Func<char, bool> predicate)
    {
        var result = "";
        while (!IsEnd() && !predicate(PeekChar()))
        {
            result += NextChar();
        }

        return result;
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

    protected string PreviousWord()
    {
        return _previousWord;
    }

    protected bool IsWordChar(char c)
    {
        return char.IsLetterOrDigit(c) || c == '_' || c == '@' || c == '#' || c == '$';
    }
}

public class TextSpan
{
    public string Word { get; set; } = string.Empty;
    public int Offset { get; set; }
    public int Length { get; set; }
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
                new ParseError($"Expected CREATE TABLE, but got {PreviousWord()} {PeekKeyword().Word}"));
        }

        return new Either<CreateTableStatement, ParseError>(new CreateTableStatement());
    }
}