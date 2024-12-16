using T1.SqlSharp;

namespace SqlSharpLit.Common.ParserLit;

public class StringParser
{
    private readonly string _text;
    private int _position;
    private TextSpan _previousWord = new();
    private static readonly char[] Brackets = ['(', ')', '{', '}', '[', ']'];

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

    public string GetPreviousText(int offset)
    {
        return _text.Substring(offset, _position - offset);
    }
    
    public string GetText(int startOffset, int endOffset)
    {
        return _text.Substring(startOffset, endOffset - startOffset);
    }

    public bool IsEnd()
    {
        return _position >= _text.Length;
    }

    public bool IsPeekIgnoreCase(Func<TextSpan> readFn, string expected)
    {
        var startPosition = _position;
        var textSpan = readFn();
        _position = startPosition;
        return IsSameIgnoreCase(textSpan.Word, expected);
    }

    public bool IsWordChar(char c)
    {
        return char.IsLetterOrDigit(c) || c == '_' || c == '@' || c == '#' || c == '$';
    }

    public TextSpan MatchSymbol(string expected)
    {
        SkipWhitespace();
        var startPosition = _position;
        foreach (char c in expected)
        {
            if (IsEnd())
            {
                throw new Exception($"Expected '{expected}' at position {_position}, but found EOF.");
            }
            var nextChar = NextChar();
            if (nextChar != c)
            {
                throw new Exception($"Expected '{expected}' at position {_position}, but found different content '{nextChar}'.");
            }
        }
        return new TextSpan
        {
            Word = _text.Substring(startPosition, _position - startPosition),
            Offset = startPosition,
            Length = _position - startPosition
        };
    }

    public char NextChar()
    {
        if (IsEnd()) return '\0';
        return _text[_position++];
    }

    public TextSpan Peek(Func<TextSpan> readFunc)
    {
        var tempPosition = _position;
        var textSpan = readFunc();
        _position = tempPosition;
        return textSpan;
    }

    public char PeekChar()
    {
        SkipWhitespace();
        if (IsEnd()) return '\0';
        return _text[_position];
    }

    public TextSpan PeekIdentifier(string word)
    {
        var span = Peek(ReadIdentifier);
        if (span.Word == word)
        {
            return span;
        }

        return new TextSpan
        {
            Word = string.Empty,
            Offset = _position,
            Length = 0
        };
    }

    public bool PeekMatchSymbol(string symbol)
    {
        var tempPosition = _position;
        var isSymbol = NextText(symbol.Length).Word == symbol;
        _position = tempPosition;
        return isSymbol;
    }

    public char PeekNext()
    {
        if (IsEnd()) return '\0';
        return _text[_position];
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

    public TextSpan PreviousWord()
    {
        return _previousWord;
    }

    public char ReadChar()
    {
        SkipWhitespace();
        if (IsEnd()) return '\0';
        return _text[_position++];
    }

    public TextSpan ReadDoubleComment()
    {
        var startPosition = _position;
        if (Try(ReadSymbols, out var openSymbol))
        {
            if (openSymbol.Word == "/*")
            {
                ReadUntil("*/");
                NextText(2);
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

    public TextSpan ReadFloat()
    {
        SkipWhitespace();
        var startOffset = _position;
        if (!Try(ReadInt, out var number))
        {
            return new TextSpan
            {
                Word = string.Empty,
                Offset = _position,
                Length = 0
            };
        }
        var dot = NextChar();
        if (dot != '.')
        {
            _position = startOffset;
            return new TextSpan
            {
                Word = string.Empty,
                Offset = _position,
                Length = 0
            };
        }
        ReadInt();
        return new TextSpan
        {
            Word = _text.Substring(startOffset, _position - startOffset),
            Offset = startOffset,
            Length = _position - startOffset
        };
    }

    public TextSpan ReadFullQuotedIdentifier()
    {
        SkipWhitespace();
        var startPosition = _position;
        var prevToken = TextSpan.None;
        while (!IsEnd())
        {
            var identifier = Or(ReadIdentifier, ReadQuotedIdentifier);
            if (identifier.Length == 0)
            {
                if (prevToken == TextSpan.None)
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

            prevToken = identifier;
            var dotdot = Peek(()=>NextText(2));
            if(dotdot.Word == "..")
            {
                NextText(2);
                prevToken = new TextSpan
                {
                    Word = dotdot.Word,
                    Offset = dotdot.Offset,
                    Length = 2
                };
                continue;
            }
            
            if(dotdot.Word == ".*")
            {
                NextText(2);
                break;
            }
            
            if (PeekNext() != '.')
            {
                break;
            }

            var dot = NextChar();
            prevToken = new TextSpan()
            {
                Word = dot.ToString(),
                Offset = _position - 1,
                Length = 1
            };
        }

        var lastPosition = _position;
        if (prevToken.Word == ".")
        {
            lastPosition = prevToken.Offset;
        }

        return new TextSpan
        {
            Word = _text.Substring(startPosition, lastPosition - startPosition),
            Offset = startPosition,
            Length = lastPosition - startPosition
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
        if (identifyPrev.Contains(identifier))
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

    public TextSpan ReadInt()
    {
        SkipWhitespace();
        var startOffset = _position;
        var ch = PeekChar();
        if (!char.IsDigit(ch))
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
            ch = NextChar();
            if (!char.IsDigit(ch))
            {
                _position--;
                break;
            }
        }

        return new TextSpan()
        {
            Word = _text.Substring(startOffset, _position - startOffset),
            Offset = startOffset,
            Length = _position - startOffset
        };
    }

    public TextSpan ReadNegativeNumber()
    {
        SkipWhitespace();
        if (PeekNext() != '-')
        {
            return new TextSpan
            {
                Word = string.Empty,
                Offset = _position,
                Length = 0
            };
        }
        var startPosition = _position;
        NextChar();
        var floatNumber = ReadFloat();
        if (floatNumber.Length != 0)
        {
            return new TextSpan
            {
                Word = _text.Substring(startPosition, _position - startPosition),
                Offset = startPosition,
                Length = 0
            };
        }
        ReadInt();
        return new TextSpan
        {
            Word = _text.Substring(startPosition, _position - startPosition),
            Offset = startPosition,
            Length = _position - startPosition
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

    public TextSpan ReadSqlDate()
    {
        var startPosition = _position;
        var year = ReadInt();
        NextChar();
        var month = ReadInt();
        NextChar();
        var day = ReadInt();
        if (year.Length == 0 || month.Length == 0 || day.Length == 0)
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

    public TextSpan ReadSqlIdentifier()
    {
        if (Try(ReadFullQuotedIdentifier, out var fullQuotedIdentifier))
        {
            return fullQuotedIdentifier;
        }
        
        if (Try(ReadIdentifier, out var identifier))
        {
            return identifier;
        }

        return new TextSpan
        {
            Word = string.Empty,
            Offset = _position,
            Length = 0
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

        var startPosition = _position;
        var startChar = ReadChar();
        if (startChar == 'N')
        {
            quoteChar = NextChar();
            if( quoteChar != '\'' && quoteChar != '"' && quoteChar != '`')
            {
                _position = startPosition;
                return new TextSpan
                {
                    Word = string.Empty,
                    Offset = startPosition,
                    Length = 0
                };
            }
        }

        while (!IsEnd())
        {
            var c = NextChar();
            if (c == quoteChar && PeekNext() == quoteChar)
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
            Word = _text.Substring(startPosition, _position - startPosition),
            Offset = startPosition,
            Length = _position - startPosition
        };
    }

    public TextSpan ReadSqlSingleComment()
    {
        var startPosition = _position;
        ReadChar();
        NextChar();
        ReadUntil(c => c == '\n');
        NextChar();
        return new TextSpan()
        {
            Word = _text.Substring(startPosition, _position - startPosition),
            Offset = startPosition,
            Length = _position - startPosition
        };
    }

    public TextSpan NextText(int length)
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

    public TextSpan ReadBracket()
    {
        SkipWhitespace();
        if (IsEnd())
        {
            return new TextSpan
            {
                Word = string.Empty,
                Offset = _position,
                Length = 0
            };
        }
        var startPosition = _position;
        var bracketStr = NextText(1).Word;
        if (!Brackets.Contains(bracketStr[0]))
        {
            return new TextSpan
            {
                Word = string.Empty,
                Offset = startPosition,
                Length = 0
            };
        }
        return new TextSpan
        {
            Word = bracketStr,
            Offset = startPosition,
            Length = 1
        };
    }

    public TextSpan ReadSymbol(int length)
    {
        SkipWhitespace();
        var startPosition = _position;
        var symbol = NextText(length);
        // if (!IsSymbolEnd(PeekNext()))
        // {
        //     return new TextSpan
        //     {
        //         Word = string.Empty,
        //         Offset = startPosition,
        //         Length = 0
        //     };
        // }
        return symbol;
    }
    
    private bool IsSymbolEnd(char symbolEnd)
    {
        if(symbolEnd == '\0')
        {
            return true;
        }
        if (IsWordChar(symbolEnd))
        {
            return true;
        }
        if (char.IsWhiteSpace(symbolEnd))
        {
            return true;
        }
        if (symbolEnd == '\'')
        {
            return true;
        }
        return Brackets.Contains(symbolEnd);
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
        while (!IsEnd() && !predicate(PeekNext()))
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
        while (!IsEnd() && !PeekString(text.Length).SequenceEqual(text))
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
            if (openSymbol.Word == "/**/")
            {
                return true;
            }
            
            if (openSymbol.Word == "/*")
            {
                _position = startPosition;
                ReadDoubleComment();
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
            if (openSymbol.Word.StartsWith("--"))
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
        _previousWord = textSpan;
        return true;
    }

    public bool TryMatch(string keyword, out TextSpan textSpan)
    {
        SkipWhitespace();
        var startPosition = _position;
        if (Try(() => NextText(keyword.Length), out var textSpan1))
        {
            if (textSpan1.Word == keyword)
            {
                textSpan = textSpan1;
                return true;
            }
            _position = startPosition;   
        }
        textSpan = new TextSpan
        {
            Word = string.Empty,
            Offset = startPosition,
            Length = 0
        };
        return false;
    }

    public bool TryMatches(params string[] keywords)
    {
        SkipWhitespace();
        var startPosition = _position;
        foreach (var keyword in keywords)
        {
            if (!TryMatch(keyword, out _))
            {
                _position = startPosition;
                return false;
            }
        }
        return true;
    }

    public bool TryKeywordsIgnoreCase(string[] keywords, out TextSpan textSpan)
    {
        SkipWhitespace();
        var startPosition = _position;
        foreach (var keyword in keywords)
        {
            if (!TryKeywordIgnoreCase(keyword, out _))
            {
                _position = startPosition;
                textSpan = TextSpan.Empty(startPosition);
                return false;
            }
        }
        textSpan = new TextSpan
        {
            Word = _text.Substring(startPosition, _position - startPosition),
            Offset = startPosition,
            Length = _position - startPosition
        };
        return true;
    }

    public bool TryKeywordIgnoreCase(string keyword, out TextSpan textSpan)
    {
        SkipWhitespace();
        var startPosition = _position;
        var readCount = 0;
        while (!IsEnd() && readCount < keyword.Length)
        {
            readCount++;
            _position++;
        }

        var word = _text.Substring(startPosition, readCount);
        if (!string.Equals(word, keyword, StringComparison.OrdinalIgnoreCase))
        {
            textSpan = TextSpan.Empty(startPosition);
            _position = startPosition;
            return false;
        }
        
        var nextChar = PeekNext();
        if (IsWordChar(nextChar))
        {
            _position = startPosition;
            textSpan = TextSpan.Empty(startPosition);
            return false;
        }
        
        _previousWord = new TextSpan
        {
            Word = word,
            Offset = _position,
            Length = keyword.Length
        };
        textSpan = new TextSpan
        {
            Word = word,
            Offset = startPosition,
            Length = readCount
        };
        return true;
    }

    private static bool IsSameIgnoreCase(string word1, string word2)
    {
        return string.Equals(word1, word2, StringComparison.OrdinalIgnoreCase);
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

    private ReadOnlySpan<char> PeekString(int length)
    {
        if (IsEnd())
        {
            return string.Empty;
        }
        var remainLength = _text.Length - _position;
        var readLength = Math.Min(length, remainLength);
        return _text.AsSpan(_position, readLength);
    }

    public TextSpan CreateSpan(TextSpan startSpan)
    {
        return new TextSpan
        {
            Word = _text.Substring(startSpan.Offset, _position - startSpan.Offset),
            Offset = startSpan.Offset,
            Length = _position - startSpan.Offset
        };
    }
    
    public TextSpan CreateSpan(int startOffset)
    {
        return new TextSpan
        {
            Word = _text.Substring(startOffset, _position - startOffset),
            Offset = startOffset,
            Length = _position - startOffset
        };
    }
}