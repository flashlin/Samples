namespace T1.ParserKit.Core.Parsers
{
    public static class ParserExtensions
    {
        #region When

        public static Parser<TInput, TValue> When<TInput, TValue>(
            this Parser<TInput, TValue> parser,
            Func<TInput, bool> predicate)
        {
            return new Parser<TInput, TValue, ParserWhenCommand<TInput, TValue>>(
                new ParserWhenCommand<TInput, TValue>(parser, predicate));
        }

        public static Parser<TInput, TValue> When<TInput, TValue, TPredicate>(

            this Parser<TInput, TValue> parser,
            TPredicate predicate)
            where TPredicate : IParsePredicate<TInput>
        {
            return new Parser<TInput, TValue, ParserWhenPredicateCommand<TInput, TValue, TPredicate>>(
                new ParserWhenPredicateCommand<TInput, TValue, TPredicate>(parser, predicate));
        }

        #endregion //When

        #region Where

        public static Parser<TInput, TValue> Where<TInput, TValue>(
            this Parser<TInput, TValue> parser, 
            Func<TValue, bool> predicate)
        {
            
            return new Parser<TInput, TValue, ParserWhereCommand<TInput, TValue>>(
                new ParserWhereCommand<TInput, TValue>(parser, predicate));
        }

        #endregion //Where

        #region Select

        public static Parser<TInput, TSecondValue> Select<TInput, TValue, TSecondValue>(
            this Parser<TInput, TValue> parser, 
            Func<TValue, TSecondValue> selector)
        {
            return new Parser<TInput, TSecondValue, ParserSelectCommand<TInput, TSecondValue, TValue>>(
                new ParserSelectCommand<TInput, TSecondValue, TValue>(parser, selector));
        }

        #endregion //Select

        #region SelectMany

        public static Parser<TInput, TSecondValue> SelectMany<TInput, TValue, TIntermediate, TSecondValue>(
            this Parser<TInput, TValue> parser,
            Func<TValue, Parser<TInput, TIntermediate>> selector,
            Func<TValue, TIntermediate, TSecondValue> projector)
        {
            return new Parser<TInput, TSecondValue, ParserSelectManyCommand<TInput, TValue, TIntermediate, TSecondValue>>(
                new ParserSelectManyCommand<TInput, TValue, TIntermediate, TSecondValue>(parser, selector, projector));
        }

        #endregion //SelectMany

        #region ZeroOrMore

        public static Parser<TInput, IEnumerable<TValue>> ZeroOrMore<TInput, TValue>(this Parser<TInput, TValue> parser)
        {
            return new Parser<TInput, IEnumerable<TValue>, ParserZeroOrMoreCommand<TInput, TValue>>(
                new ParserZeroOrMoreCommand<TInput, TValue>(parser));
        }

        #endregion //ZeroOrMore

        #region OneOrMore

        public static Parser<TInput, IEnumerable<TValue>> OneOrMore<TInput, TValue>(this Parser<TInput, TValue> parser)
        {
            return new Parser<TInput, IEnumerable<TValue>, ParserOneOrMoreCommand<TInput, TValue>>(
                new ParserOneOrMoreCommand<TInput, TValue>(parser));
        }

        #endregion //OneOrMore

        #region Optional

        public static Parser<TInput, IEnumerable<TValue>> Optional<TInput, TValue>(this Parser<TInput, TValue> parser)
        {
            return new Parser<TInput, IEnumerable<TValue>, ParserOptionalCommand<TInput, TValue>>(
                new ParserOptionalCommand<TInput, TValue>(parser));
        }

        #endregion //Optional
    }

    #region ParserWhenCommand

    internal struct ParserWhenCommand<TInput, TValue> : IParseCommand<TInput, TValue>
        {
        private Parser<TInput, TValue> parser;
        private Func<TInput, bool> predicate;

        public ParserWhenCommand(Parser<TInput, TValue> parser, Func<TInput, bool> predicate)
        {
            this.parser = parser;
            this.predicate = predicate;
        }

        public ParseResult<TInput, TValue> Execute(TInput input)
        {
            if (!predicate(input))
                return null;

            return parser.Parse(input);
        }
    }

    #endregion //ParserWhenCommand

    #region ParserWhereCommand

    internal struct ParserWhereCommand<TInput, TValue> : IParseCommand<TInput, TValue>
    {
        private Parser<TInput, TValue> parser;
        private Func<TValue, bool> predicate;

        public ParserWhereCommand(Parser<TInput, TValue> parser, Func<TValue, bool> predicate)
        {
            this.parser = parser;
            this.predicate = predicate;
        }

        public ParseResult<TInput, TValue> Execute(TInput input)
        {
            var result = this.parser.Parse(input);
            if (result == null || !this.predicate(result.Value))
                return null;

            return result;
        }
    }

    #endregion //ParserWhereCommand

    internal struct ParserSelectCommand<TInput, TSecondValue, TValue> : IParseCommand<TInput, TSecondValue>
    {
        private Parser<TInput, TValue> parser;
        private Func<TValue, TSecondValue> selector;

        public ParserSelectCommand(Parser<TInput, TValue> parser, Func<TValue, TSecondValue> selector)
        {
            this.parser = parser;
            this.selector = selector;
        }

        public ParseResult<TInput, TSecondValue> Execute(TInput input)
        {
            var result = parser.Parse(input);
            if (result == null)
                return null;

            return new ParseResult<TInput, TSecondValue>(selector(result.Value), result.Rest);
        }
    }

    internal struct ParserSelectManyCommand<TInput, TValue, TIntermediate, TSecondValue> : IParseCommand<TInput, TSecondValue>
    {
        private Parser<TInput, TValue> parser;
        private Func<TValue, Parser<TInput, TIntermediate>> selector;
        private Func<TValue, TIntermediate, TSecondValue> projector;

        public ParserSelectManyCommand(Parser<TInput, TValue> parser, Func<TValue, Parser<TInput, TIntermediate>> selector, Func<TValue, TIntermediate, TSecondValue> projector)
        {
            this.parser = parser;
            this.selector = selector;
            this.projector = projector;
        }

        public ParseResult<TInput, TSecondValue> Execute(TInput input)
        {
            var result = parser.Parse(input);
            if (result == null)
                return null;

            var value = result.Value;
            var secondResult = selector(value).Parse(result.Rest);
            if (secondResult == null)
                return null;

            return new ParseResult<TInput, TSecondValue>(projector(value, secondResult.Value), secondResult.Rest);
        }
    }

    internal struct ParserWhenPredicateCommand<TInput, TValue, TPredicate> : IParseCommand<TInput, TValue>
        where TPredicate : IParsePredicate<TInput>
    {
        private Parser<TInput, TValue> parser;
        private TPredicate predicate;

        public ParserWhenPredicateCommand(Parser<TInput, TValue> parser, TPredicate predicate)
        {
            this.parser = parser;
            this.predicate = predicate;
        }

        public ParseResult<TInput, TValue> Execute(TInput input)
        {
            if (!predicate.Execute(input))
                return null;

            return parser.Parse(input);
        }
    }

    internal struct ParserZeroOrMoreCommand<TInput, TValue> : IParseCommand<TInput, IEnumerable<TValue>>
    {
        private Parser<TInput, TValue> parser;

        public ParserZeroOrMoreCommand(Parser<TInput, TValue> parser)
        {
            this.parser = parser;
        }

        public ParseResult<TInput, IEnumerable<TValue>> Execute(TInput input)
        {
            ParseResult<TInput, TValue> result;
            var value = new List<TValue>();
            var rest = input;
            do
            {
                result = this.parser.Parse(rest);
                if (result != null)
                {
                    value.Add(result.Value);
                    rest = result.Rest;
                }
            }
            while (result != null);

            return new ParseResult<TInput, IEnumerable<TValue>>(value, rest);
        }
    }

    #region ParserOneOrMoreCommand

    internal struct ParserOneOrMoreCommand<TInput, TValue> : IParseCommand<TInput, IEnumerable<TValue>>
    {
        private Parser<TInput, TValue> parser;

        public ParserOneOrMoreCommand(Parser<TInput, TValue> parser)
        {
            this.parser = parser;
        }

        public ParseResult<TInput, IEnumerable<TValue>> Execute(TInput input)
        {
            var result = this.parser.Parse(input);
            if (result == null)
                return null;

            var zeroOrMoreTextParser = new Parser<TInput, IEnumerable<TValue>, ParserZeroOrMoreCommand<TInput, TValue>>(
                new ParserZeroOrMoreCommand<TInput, TValue>(this.parser));

            var secondResult = zeroOrMoreTextParser.Parse(result.Rest);
            if (secondResult == null)
                return new ParseResult<TInput, IEnumerable<TValue>>(new[] { result.Value }, result.Rest);

            return new ParseResult<TInput, IEnumerable<TValue>>(
                new[] { result.Value }.Concat(secondResult.Value),
                secondResult.Rest);
        }
    }

    #endregion //ParserOneOrMoreCommand

    #region ParserOptionalCommand

    internal struct ParserOptionalCommand<TInput, TValue> : IParseCommand<TInput, IEnumerable<TValue>>
    {
        private Parser<TInput, TValue> parser;

        public ParserOptionalCommand(Parser<TInput, TValue> parser)
        {
            this.parser = parser;
        }

        public ParseResult<TInput, IEnumerable<TValue>> Execute(TInput input)
        {
            var result = parser.Parse(input);
            if (result == null)
                return new ParseResult<TInput, IEnumerable<TValue>>(new TValue[0], input);

            return new ParseResult<TInput, IEnumerable<TValue>>(new[] { result.Value }, result.Rest);
        }
    }

    #endregion //ParserOptionalCommand
}
