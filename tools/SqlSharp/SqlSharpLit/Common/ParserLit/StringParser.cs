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

    public int Position => _position;

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

    public bool IsPeekIdentifier(string word)
    {
        return PeekIdentifier(word).Length != 0;
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

    public string NextString(int length)
    {
        if (IsEnd()) return string.Empty;
        var text = PeekString(length);
        _position += text.Length;
        return text;
    }

    public char PeekChar()
    {
        if (IsEnd()) return '\0';
        SkipWhitespace();
        return _text[_position];
    }

    public TextSpan PeekIdentifier(string word)
    {
        SkipWhitespace();
        var tempPosition = _position;
        if (Try(ReadIdentifier, out var identifier))
        {
            _position = tempPosition;
            if (identifier.Word == word)
            {
                return identifier;
            }
        }

        _position = tempPosition;
        return new TextSpan
        {
            Word = string.Empty,
            Offset = _position,
            Length = 0
        };
    }

    public TextSpan PeekWord()
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

    public string PeekString(int length)
    {
        if (IsEnd()) return string.Empty;
        var remainLength = _text.Length - _position;
        var readLength = Math.Min(length, remainLength);
        return _text.Substring(_position, readLength);
    }

    public TextSpan PreviousWord()
    {
        return _previousWord;
    }

    public TextSpan ReadFullQuotedIdentifier()
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
        ReadQuotedIdentifier();
        while (!IsEnd())
        {
            var c = NextChar();
            if (c != '.')
            {
                _position--;
                break;
            }

            ReadQuotedIdentifier();
        }

        return new TextSpan()
        {
            Word = _text.Substring(offset, _position - offset),
            Offset = offset,
            Length = _position - offset
        };
    }
    
    public TextSpan ReadIdentifier()
    {
        SkipWhitespace();
        var offset = _position;
        var ch = PeekChar();
        if (!char.IsLetter(ch) && ch != '_' && ch != '@' && ch != '#' && ch != '$')
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

        if (identifier.Length == 1)
        {
            return new TextSpan
            {
                Word = string.Empty,
                Offset = _position,
                Length = 0
            };
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

    public TextSpan ReadSqlDoubleComment()
    {
        var startPosition = _position;
        if (Try(ReadSymbol, out var openSymbol))
        {
            if (openSymbol.Word == "/*")
            {
                _position = startPosition;
                ReadUntil("*/");
                NextString(2);
                return new TextSpan
                {
                    Word = _text.Substring(startPosition, _position - startPosition),
                    Offset = startPosition,
                    Length = _position - startPosition
                };
            }
        }
        _position = startPosition;
        return new TextSpan
        {
            Word = string.Empty,
            Offset = startPosition,
            Length = 0
        };
    }

    public TextSpan ReadSqlIdentifier()
    {
        if (Try(ReadIdentifier, out var identifier))
        {
            return identifier;
        }

        if (Try(ReadFullQuotedIdentifier, out var fullQuotedIdentifier))
        {
            return fullQuotedIdentifier;
        }

        return new TextSpan
        {
            Word = string.Empty,
            Offset = _position,
            Length = 0
        };
    }

    public TextSpan ReadSqlSingleComment()
    {
        var startPosition = _position;
        NextChar();
        NextChar();
        ReadUntil(c => c == '\n');
        return new TextSpan()
        {
            Word = _text.Substring(startPosition, _position - startPosition),
            Offset = startPosition,
            Length = _position - startPosition
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

    public TextSpan ReadUntil(string text)
    {
        var offset = _position;
        var result = "";
        while (!IsEnd() && PeekString(text.Length)!=text)
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

    public TextSpan ReadUntilRightParenthesis()
    {
        var startPosition = _position;
        var openParenthesis = 0;
        while (!IsEnd())
        {
            var c = NextChar();
            if (c == '(')
            {
                openParenthesis++;
                continue;
            }
            if (c == ')')
            {
                openParenthesis--;
                if (openParenthesis == -1)
                {
                    _position--;
                    return new TextSpan()
                    {
                        Word = _text.Substring(startPosition, _position - startPosition),
                        Offset = startPosition,
                        Length = _position - startPosition
                    };
                }
            }
        }
        _position = startPosition;
        return new TextSpan()
        {
            Word = string.Empty,
            Offset = startPosition,
            Length = 0
        };
    }

    public void SkipSqlComment()
    {
        SkipSqlDoubleComment();
        SkipSqlSingleComment();
    }

    public void SkipSqlDoubleComment()
    {
        var startPosition = _position;
        if (Try(ReadSymbol, out var openSymbol))
        {
            if (openSymbol.Word == "/*")
            {
                _position = startPosition;
                ReadSqlDoubleComment();
                return;
            }
        }
        _position = startPosition;
    }

    public void SkipSqlSingleComment()
    {
        var startPosition = _position;
        if (Try(ReadSymbol, out var openSymbol))
        {
            if (openSymbol.Word == "--")
            {
                _position = startPosition;
                ReadSqlSingleComment();
                return;
            }
        }
        _position = startPosition;
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
    
    
    public bool TryMatchIgnoreCase(string keyword)
    {
        SkipWhitespace();
        var tempPosition = _position;
        var word = "";
        while (tempPosition < _text.Length && word.Length < keyword.Length)
        {
            word += _text[tempPosition];
            tempPosition++;
        }

        if (!string.Equals(word, keyword, StringComparison.OrdinalIgnoreCase))
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

    public bool TryMatch(string keyword)
    {
        SkipWhitespace();
        var tempPosition = _position;
        var word = "";
        while (tempPosition < _text.Length && word.Length < keyword.Length)
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

    public bool TryMatches(params string[] keywords)
    {
        SkipWhitespace();
        var tempPosition = _position;
        foreach (var keyword in keywords)
        {
            if (!TryMatch(keyword))
            {
                _position = tempPosition;
                return false;
            }
        }
        return true;
    }
    
    public bool TryMatchesIgnoreCase(params string[] keywords)
    {
        SkipWhitespace();
        var tempPosition = _position;
        foreach (var keyword in keywords)
        {
            if (!TryMatchIgnoreCase(keyword))
            {
                _position = tempPosition;
                return false;
            }
        }
        return true;
    }

    public bool TryMatchIgnoreCaseKeyword(string keyword)
    {
        var peek = PeekWord();
        if (peek.Word != keyword.ToUpper()) return false;
        _previousWord = peek;
        _position = peek.Offset + peek.Length;
        return true;
    }
}