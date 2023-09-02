using T1.ParserKit.Core.Parsers;

namespace T1.ParserKit.Core
{
    public abstract class Grammar<TOutput>
    {
        public abstract Parser<TextUtils.Text, TOutput> Parser { get; }
        public TextParser String(string text)
        {
            return new TextParser<StringParseCommand>(
                new StringParseCommand(text));
        }

        public SetParser Char()
        {
            return new SetParser<AnyCharParseCommand>(
                new AnyCharParseCommand());
        }

        public SetParser Char(char character)
        {
            return new SetParser<CharParseCommand>(
                new CharParseCommand(character));
        }


        public SetParser Set(string set)
        {
            return new SetParser<SetParseCommand>(
                new SetParseCommand(set));
        }


        public SetParser Range(char from, char to)
        {
            return new SetParser<RangeParseCommand>(
                new RangeParseCommand(from, to));
        }


        private TextParser? _nullParser;
        public TextParser Null
        {
            get
            {
                if (_nullParser == null)
                    _nullParser = new TextParser<NullTextParseCommand>(new NullTextParseCommand());
                return _nullParser;
            }
        }


        private SetParser? _singleWhiteSpace;
        public SetParser SingleWhiteSpace
        {
            get
            {
                if (_singleWhiteSpace == null)
                {
                    _singleWhiteSpace = Set(" \f\n\r\t\v");
                }
                return _singleWhiteSpace;
            }
        }


        private TextParser? _whiteSpace;
        public TextParser WhiteSpace
        {
            get
            {
                if (_whiteSpace == null)
                {
                    _whiteSpace = SingleWhiteSpace.ZeroOrMore();
                }
                return _whiteSpace;
            }
        }

        private TextParser? _digit;
        public TextParser Digit
        {
            get
            {
                if (this._digit == null)
                    this._digit = Range('0', '9');
                return this._digit;
            }
        }


        private TextParser? _letter;
        public TextParser Letter
        {
            get
            {
                if (this._letter == null)
                    this._letter = Range('a', 'z') | Range('A', 'Z');
                return this._letter;
            }
        }
    }


    internal struct NullTextParseCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        public ParseResult<TextUtils.Text, TextUtils.Text>? Execute(TextUtils.Text input)
        {
            return null;
        }
    }


    internal readonly struct StringParseCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private readonly string _text;

        public StringParseCommand(string text)
        {
            this._text = text;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text>? Execute(TextUtils.Text input)
        {
            if (input.Length < this._text.Length)
                return null;

            for (var i = 0; i < this._text.Length; i++)
                if (_text[i] != input[i])
                    return null;

            var split = input.Split(this._text.Length);
            return new ParseResult<TextUtils.Text, TextUtils.Text>(split.Head, split.Tail);
        }
    }



    internal struct AnyCharParseCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        public ParseResult<TextUtils.Text, TextUtils.Text>? Execute(TextUtils.Text input)
        {
            if (input.Length == 0)
                return null;
            var split = input.Split(1);
            return new ParseResult<TextUtils.Text, TextUtils.Text>(split.Head, split.Tail);
        }
    }


    internal readonly struct CharParseCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private readonly char character;

        public CharParseCommand(char character)
        {
            this.character = character;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text>? Execute(TextUtils.Text input)
        {
            if (input.Length == 0 || input[0] != this.character)
                return null;

            var split = input.Split(1);
            return new ParseResult<TextUtils.Text, TextUtils.Text>(split.Head, split.Tail);
        }
    }

    internal readonly struct SetParseCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private readonly string _set;

        public SetParseCommand(string set)
        {
            this._set = set;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text>? Execute(TextUtils.Text input)
        {
            if (input.Length == 0)
                return null;

            var character = input[0];
            var inSet = false;

            for (var i = 0; i < _set.Length; i++)
            {
                if (_set[i] == character)
                {
                    inSet = true;
                    break;
                }
            }

            if (!inSet)
                return null;

            var split = input.Split(1);
            return new ParseResult<TextUtils.Text, TextUtils.Text>(split.Head, split.Tail);
        }
    }



    internal struct RangeParseCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private readonly char _from;
        private readonly char _to;

        public RangeParseCommand(char from, char to)
        {
            this._from = from;
            this._to = to;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text>? Execute(TextUtils.Text input)
        {
            if (input.Length == 0)
                return null;

            var characterNumber = (int)input[0];
            var inSet = false;

            var fromNumber = (int)_from;
            var toNumber = (int)_to;
            if (fromNumber > toNumber)
            {
                (fromNumber, toNumber) = (toNumber, fromNumber);
            }

            for (var currentCharacterNumber = fromNumber; currentCharacterNumber <= toNumber; currentCharacterNumber++)
            {
                if (currentCharacterNumber == characterNumber)
                {
                    inSet = true;
                    break;
                }
            }

            if (!inSet)
                return null;

            var split = input.Split(1);
            return new ParseResult<TextUtils.Text, TextUtils.Text>(split.Head, split.Tail);
        }
    }

}