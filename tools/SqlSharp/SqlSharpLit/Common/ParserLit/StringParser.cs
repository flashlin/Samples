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
        if (!PeekKeyword(keyword)) return false;
        var word = "";
        while (!IsEnd() && IsWordChar(PeekChar()))
        {
            word += NextChar();
        }
        _previousWord = word;
        return true;
    }

    protected bool PeekKeyword(string keyword)
    {
        SkipWhitespace();
        var tempPosition = _position;
        var keywordUpper = keyword.ToUpper();
        var word = "";

        // 收集完整的單詞
        while (tempPosition < _text.Length && IsWordChar(_text[tempPosition]))
        {
            word += _text[tempPosition];
            tempPosition++;
        }

        // 檢查是否完全匹配（大小寫不敏感）
        if (word.ToUpper() == keywordUpper)
        {
            // 確保後面是空白字符或特殊字符
            if (tempPosition >= _text.Length || 
                char.IsWhiteSpace(_text[tempPosition]) || 
                _text[tempPosition] == '(' || 
                _text[tempPosition] == ')' || 
                _text[tempPosition] == ',' ||
                _text[tempPosition] == '.')
            {
                return true;
            }
        }

        return false;
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

public class SqlParser : StringParser
{
    public SqlParser(string text) : base(text)
    {
    }
    
    public CreateTableStatement ParseCreateTableStatement()
    {
        if( !(TryMatchKeyword("CREATE") && TryMatchKeyword("TABLE")) )
        {
            throw new Exception("Expected CREATE TABLE");
        }

        return new CreateTableStatement();
    }
}