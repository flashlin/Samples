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

    public int Position
    {
        get => _position;
        set => _position = value;
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

    public bool IsPeekIdentifier(string word)
    {
        return PeekIdentifier(word).Length != 0;
    }
    
    public bool IsPeekIdentifiers(params string[] words)
    {
        foreach (var word in words)
        {
            if (IsPeekIdentifier(word))
            {
                return true;
            }
        }
        return false;
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
            if (IsEnd() || char.ToUpper(ReadChar()) != char.ToUpper(c))
            {
                throw new Exception($"Expected '{expected}' at position {_position}, but found different content");
            }
        }
    }

    public char ReadChar()
    {
        SkipWhitespace();
        if (IsEnd()) return '\0';
        return _text[_position++];
    }

    public char NextChar()
    {
        if (IsEnd()) return '\0';
        return _text[_position++];
    }

    public ReadOnlySpan<char> NextString(int length)
    {
        if (IsEnd()) return string.Empty;
        var text = PeekString(length);
        _position += text.Length;
        return text;
    }

    public char PeekChar()
    {
        SkipWhitespace();
        if (IsEnd()) return '\0';
        return _text[_position];
    }

    public char Peek()
    {
        if (IsEnd()) return '\0';
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
        while (tempPosition < _text.Length && IsWordChar(_text[tempPosition]))
        {
            tempPosition++;
        }

        return new TextSpan
        {
            Word = _text.Substring(_position, tempPosition - _position),
            Offset = _position,
            Length = tempPosition - _position
        };
    }

    public ReadOnlySpan<char> PeekString(int length)
    {
        if (IsEnd()) return string.Empty;
        var remainLength = _text.Length - _position;
        var readLength = Math.Min(length, remainLength);
        return _text.AsSpan(_position, readLength);
    }

    public TextSpan PreviousWord()
    {
        return _previousWord;
    }

    private TextSpan Or(params Func<TextSpan>[] readFnList)
    {
        foreach (var readFn in readFnList)
        {
            var textSpan = readFn();
            if (textSpan.Length != 0)
            {
                return textSpan;
            }
        }
        return new TextSpan()
        {
            Word = string.Empty,
            Offset = _position,
            Length = 0
        };
    }

    public TextSpan ReadFullQuotedIdentifier()
    {
        SkipWhitespace();
        var startPosition = _position;
        var count = 0;
        while(!IsEnd())
        {
            var identifier = Or(ReadIdentifier, ReadQuotedIdentifier);
            if (identifier.Length == 0)
            {
                if (count == 0)
                {
                    return new TextSpan
                    {
                        Word = string.Empty,
                        Offset = startPosition,
                        Length = 0
                    };
                }
                break;
            }
            count++;
            
            var dot = Peek();
            if(dot!='.')
            {
                break;
            }
            NextChar();
        }
        return new TextSpan
        {
            Word = _text.Substring(startPosition, _position - startPosition),
            Offset = startPosition,
            Length = _position - startPosition
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

        while (!IsEnd())
        {
            var c = NextChar();
            if (!IsWordChar(c))
            {
                _position--;
                break;
            }
        }

        var identifyPrev = new[] { "@", "#", "$" };
        var identifier = _text.Substring(offset, _position - offset);
        if ( identifyPrev.Contains(identifier))
        {
            return new TextSpan
            {
                Word = string.Empty,
                Offset = offset,
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

    public TextSpan ReadSqlDate()
    {
        var startPosition = _position;
        var year = ReadNumber();
        NextChar();
        var month = ReadNumber();
        NextChar();
        var day = ReadNumber();
        if( year.Length ==0 || month.Length == 0 || day.Length == 0)
        {
            _position = startPosition;
            return new TextSpan
            {
                Word = string.Empty,
                Offset = startPosition,
                Length = 0
            };
        }
        return new TextSpan()
        {
            Word = _text.Substring(startPosition, _position - startPosition),
            Offset = startPosition,
            Length = _position - startPosition
        };
    }

    public TextSpan ReadSqlQuotedString()
    {
        var quoteChar = PeekChar();
        if (quoteChar != '\'' && quoteChar != '"' && quoteChar != '`' && quoteChar != 'N')
        {
            return new TextSpan()
            {
                Word = string.Empty,
                Offset = _position,
                Length = 0
            };
        }

        var offset = _position;
        var startChar = ReadChar();
        if (startChar == 'N')
        {
            quoteChar = NextChar();
        }

        while (!IsEnd())
        {
            var c = NextChar();
            if (c == quoteChar && Peek() == quoteChar)
            {
                NextChar();
                continue;
            }

            if (c == quoteChar)
            {
                break;
            }
        }

        return new TextSpan()
        {
            Word = _text.Substring(offset, _position - offset),
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
        ReadChar();
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
        if (Try(ReadSymbols, out var openSymbol))
        {
            if (openSymbol.Word == "/*")
            {
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
        ReadChar();
        ReadChar();
        ReadUntil(c => c == '\n');
        NextChar();
        return new TextSpan()
        {
            Word = _text.Substring(startPosition, _position - startPosition),
            Offset = startPosition,
            Length = _position - startPosition
        };
    }

    public bool PeekMatchSymbol(string symbol)
    {
        var tempPosition = _position;
        var isSymbol = ReadString(symbol.Length).Word == symbol;
        _position = tempPosition;
        return isSymbol;
    }

    public TextSpan ReadString(int length)
    {
        length = Math.Min(length, _text.Length - _position);
        var span = new TextSpan
        {
            Word = _text.Substring(_position, length),
            Offset = _position,
            Length = length
        };
        _position += length;
        return span;
    }
    
    public TextSpan Peek(Func<TextSpan> readFunc)
    {
        var tempPosition = _position;
        var textSpan = readFunc();
        _position = tempPosition;
        return textSpan;
    } 

    public TextSpan ReadSymbols()
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
            if (IsWordChar(c) || char.IsWhiteSpace(c))
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
        while (!IsEnd() && !predicate(Peek()))
        {
            NextChar();
        }

        return new TextSpan()
        {
            Word = _text.Substring(offset, _position - offset),
            Offset = offset,
            Length = _position - offset
        };
    }

    public TextSpan ReadUntil(string text)
    {
        var offset = _position;
        while (!IsEnd() && PeekString(text.Length).ToString() != text)
        {
            NextChar();
        }

        return new TextSpan()
        {
            Word = _text.Substring(offset, _position - offset),
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
            var c = ReadChar();
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

    public bool SkipSqlComment()
    {
        var isSkipSqlDoubleComment = SkipSqlDoubleComment();
        var isSkipSqlSingleComment = SkipSqlSingleComment();
        return isSkipSqlDoubleComment || isSkipSqlSingleComment;
    }

    public bool SkipSqlDoubleComment()
    {
        var startPosition = _position;
        if (Try(ReadSymbols, out var openSymbol))
        {
            if (openSymbol.Word == "/*")
            {
                _position = startPosition;
                ReadSqlDoubleComment();
                return true;
            }
        }

        _position = startPosition;
        return false;
    }

    public bool SkipSqlSingleComment()
    {
        var startPosition = _position;
        if (Try(ReadSymbols, out var openSymbol))
        {
            if (openSymbol.Word == "--")
            {
                _position = startPosition;
                ReadSqlSingleComment();
                return true;
            }
        }

        _position = startPosition;
        return false;
    }

    public bool SkipWhitespace()
    {
        var isSkip = false;
        while (!IsEnd() && char.IsWhiteSpace(_text[_position]))
        {
            _position++;
            isSkip = true;
        }

        return isSkip;
    }

    public bool Try(Func<TextSpan> readFunc, out TextSpan textSpan)
    {
        var startPosition = _position; 
        textSpan = readFunc();
        if (textSpan.Length == 0)
        {
            _position = startPosition;
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
        if (!string.Equals(peek.Word, keyword, StringComparison.OrdinalIgnoreCase)) return false;
        _previousWord = peek;
        _position = peek.Offset + peek.Length;
        return true;
    }
}