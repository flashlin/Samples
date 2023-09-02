namespace T1.ParserKit.Core.Parsers
{
    public abstract class TextParser : Parser<TextUtils.Text, TextUtils.Text>
    {
        public TextParser ZeroOrMore()
        {
            return new TextParser<TextParserZeroOrMoreCommand>(new TextParserZeroOrMoreCommand(this));
        }

        public TextParser OneOrMore()
        {
            return new TextParser<TextParserOneOrMoreCommand>(
                new TextParserOneOrMoreCommand(this));
        }

        public TextParser Optional()
        {
            return new TextParser<TextParserOptionalCommand>(
                new TextParserOptionalCommand(this));
        }

        public Parser<TextUtils.Text, ValueWithTrivia> WithTrivia()
        {
            return
                from leftTrivia in WhiteSpace
                from value in this
                from rightTrivia in WhiteSpace
                select new ValueWithTrivia(leftTrivia, value, rightTrivia);
        }

        public Parser<TextUtils.Text, ValueWithLeftTrivia> WithLeftTrivia()
        {
            return
                from leftTrivia in WhiteSpace
                from value in this
                select new ValueWithLeftTrivia(leftTrivia, value);
        }

        public Parser<TextUtils.Text, ValueWithRightTrivia> WithRightTrivia()
        {
            return
                from value in this
                from rightTrivia in WhiteSpace
                select new ValueWithRightTrivia(value, rightTrivia);
        }

        private static TextParser? _whiteSpace;
        internal static TextParser WhiteSpace
        {
            get
            {
                if (_whiteSpace == null)
                    _whiteSpace = new TextGrammar().WhiteSpace;
                return _whiteSpace;
            }
        }

        internal new static TextParser Insert(TextUtils.Text value)
        {
            return new TextParser<TextParserInsertCommand>(
                new TextParserInsertCommand(value));
        }

        public static TextParser operator |(TextParser first, TextParser second)
        {
            return new TextParser<TextParserOrCommand>(
                new TextParserOrCommand(first, second));
        }

        public static TextParser operator +(TextParser first, TextParser second)
        {
            return new TextParser<TextParserPlusCommand>(
                new TextParserPlusCommand(first, second));
        }

        internal class TextGrammar : Grammar<TextUtils.Text>
        {
            public override Parser<TextUtils.Text, TextUtils.Text> Parser
            {
                get
                {
                    throw new NotImplementedException();
                }
            }
        }
    }

    public class TextParser<TCommand> : TextParser
        where TCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        internal TCommand command;

        public TextParser(TCommand command)
        {
            this.command = command;
        }

        public override ParseResult<TextUtils.Text, TextUtils.Text> Parse(TextUtils.Text input)
        {
            return this.command.Execute(input);
        }
    }

    public struct ValueWithTrivia
    {
        private readonly TextUtils.Text _leftTrivia;
        private readonly TextUtils.Text _value;
        private readonly TextUtils.Text _rightTrivia;

        public TextUtils.Text LeftTrivia { get { return this._leftTrivia; } }
        public TextUtils.Text Value { get { return this._value; } }
        public TextUtils.Text RightTrivia { get { return this._rightTrivia; } }

        public ValueWithTrivia(TextUtils.Text leftTrivia, TextUtils.Text value, TextUtils.Text rightTrivia)
        {
            this._leftTrivia = leftTrivia;
            this._value = value;
            this._rightTrivia = rightTrivia;
        }
    }

    public struct ValueWithLeftTrivia
    {
        private TextUtils.Text leftTrivia;
        private TextUtils.Text value;

        public TextUtils.Text LeftTrivia { get { return this.leftTrivia; } }
        public TextUtils.Text Value { get { return this.value; } }

        public ValueWithLeftTrivia(TextUtils.Text leftTrivia, TextUtils.Text value)
        {
            this.leftTrivia = leftTrivia;
            this.value = value;
        }
    }

    public struct ValueWithRightTrivia
    {
        private TextUtils.Text value;
        private TextUtils.Text rightTrivia;
        
        public TextUtils.Text Value { get { return this.value; } }
        public TextUtils.Text RightTrivia { get { return this.rightTrivia; } }

        public ValueWithRightTrivia(TextUtils.Text value, TextUtils.Text rightTrivia)
        {
            this.value = value;
            this.rightTrivia = rightTrivia;
        }
    }

    internal struct TextParserOrCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private TextParser first;
        private TextParser second;

        public TextParserOrCommand(TextParser first, TextParser second)
        {
            this.first = first;
            this.second = second;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text> Execute(TextUtils.Text input)
        {
            return
                this.first.Parse(input) ??
                this.second.Parse(input);
        }
    }

    internal struct TextParserPlusCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private TextParser first;
        private TextParser second;

        public TextParserPlusCommand(TextParser first, TextParser second)
        {
            this.first = first;
            this.second = second;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text> Execute(TextUtils.Text input)
        {
            var firstResult = first.Parse(input);
            if (firstResult == null)
                return null;

            var secondResult = second.Parse(firstResult.Rest);
            if (secondResult == null)
                return null;

            return new ParseResult<TextUtils.Text, TextUtils.Text>(
                firstResult.Value.Append(secondResult.Value),
                secondResult.Rest);
        }
    }

    internal struct TextParserInsertCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private TextUtils.Text value;

        public TextParserInsertCommand(TextUtils.Text value)
        {
            this.value = value;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text> Execute(TextUtils.Text input)
        {
            return new ParseResult<TextUtils.Text, TextUtils.Text>(this.value, input);
        }
    }

    internal struct TextParserZeroOrMoreCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private TextParser parser;

        public TextParserZeroOrMoreCommand(TextParser parser)
        {
            this.parser = parser;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text> Execute(TextUtils.Text input)
        {
            ParseResult<TextUtils.Text, TextUtils.Text> result;
            TextUtils.Text value = TextUtils.Text.Empty;
            var rest = input;
            do
            {
                result = this.parser.Parse(rest);
                if (result != null)
                {
                    value = value.Append(result.Value);
                    rest = result.Rest;
                }
            }
            while (result != null && input.Length > result.Rest.Length);

            return new ParseResult<TextUtils.Text, TextUtils.Text>(value, rest);
        }
    }

    internal struct TextParserOneOrMoreCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private TextParser parser;

        public TextParserOneOrMoreCommand(TextParser parser)
        {
            this.parser = parser;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text> Execute(TextUtils.Text input)
        {
            var result = this.parser.Parse(input);
            if (result == null)
                return null;
            if (result.Rest.Length == input.Length)
                return result;

            var zeroOrMoreTextParser = new TextParser<TextParserZeroOrMoreCommand>(
                new TextParserZeroOrMoreCommand(this.parser));

            var secondResult = zeroOrMoreTextParser.Parse(result.Rest);
            if (secondResult == null)
                return result;

            return new ParseResult<TextUtils.Text, TextUtils.Text>(
                result.Value.Append(secondResult.Value), 
                secondResult.Rest);
        }
    }

    internal struct TextParserOptionalCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private TextParser parser;

        public TextParserOptionalCommand(TextParser parser)
        {
            this.parser = parser;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text> Execute(TextUtils.Text input)
        {
            var result = parser.Parse(input);
            if (result == null)
                return new ParseResult<TextUtils.Text, TextUtils.Text>(TextUtils.Text.Empty, input);

            return result;
        }
    }
}
