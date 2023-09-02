namespace T1.ParserKit.Core.Parsers
{
    public abstract class SetParser : TextParser
    {
        public SetParser Where(Func<TextUtils.Text, bool> predicate)
        {
            return new SetParser<SetParserWhereCommand>(
                new SetParserWhereCommand(this, predicate));
        }

        public static SetParser operator |(SetParser first, SetParser second)
        {
            return new SetParser<SetParserOrCommand>(
                new SetParserOrCommand(first, second));
        }

        public static SetParser operator !(SetParser parser)
        {
            return new SetParser<SetParserNotCommand>(
                new SetParserNotCommand(parser));
        }
    }

    public class SetParser<TCommand> : SetParser
        where TCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        internal TCommand command;

        public SetParser(TCommand command)
        {
            this.command = command;
        }

        public override ParseResult<TextUtils.Text, TextUtils.Text> Parse(TextUtils.Text value)
        {
            return this.command.Execute(value);
        }
    }

    internal struct SetParserWhereCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private SetParser parser;
        private Func<TextUtils.Text, bool> predicate;

        public SetParserWhereCommand(SetParser parser, Func<TextUtils.Text, bool> predicate)
        {
            this.parser = parser;
            this.predicate = predicate;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text> Execute(TextUtils.Text input)
        {
            var result = this.parser.Parse(input);
            if (result == null || !this.predicate(result.Value))
                return null;

            return result;
        }
    }

    internal struct SetParserOrCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private SetParser first;
        private SetParser second;

        public SetParserOrCommand(SetParser first, SetParser second)
        {
            this.first = first;
            this.second = second;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text> Execute(TextUtils.Text input)
        {
            var firstResult = first.Parse(input);
            if (firstResult != null)
                return firstResult;

            var secondResult = second.Parse(input);
            if (secondResult != null)
                return secondResult;

            return null;
        }
    }

    internal struct SetParserNotCommand : IParseCommand<TextUtils.Text, TextUtils.Text>
    {
        private SetParser parser; 

        public SetParserNotCommand(SetParser parser)
        {
            this.parser = parser;
        }

        public ParseResult<TextUtils.Text, TextUtils.Text> Execute(TextUtils.Text input)
        {
            if (input.Length == 0)
                return null;

            var result = this.parser.Parse(input);
            if (result != null)
                return null;

            var split = input.Split(1);
            return new ParseResult<TextUtils.Text, TextUtils.Text>(split.Head, split.Tail);
        }
    }
}
