using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Linq.Parsers.Testing")]
namespace T1.ParserKit.Core.Parsers
{
    public abstract class Parser<TInput, TValue>
    {
        public abstract ParseResult<TInput, TValue>? Parse(TInput value);

        public static Parser<TInput, TValue> operator |(Parser<TInput, TValue> first, Parser<TInput, TValue> second)
        {
            return new Parser<TInput, TValue, ParserOrCommand<TInput, TValue>>(
                new ParserOrCommand<TInput, TValue>(first, second));
        }

        internal static Parser<TInput, TValue> Insert(TValue value)
        {
            return new Parser<TInput, TValue, ParserInsertCommand<TInput, TValue>>(
                new ParserInsertCommand<TInput, TValue>(value));
        }
    }


    public class Parser<TInput, TValue, TCommand> : Parser<TInput, TValue>
        where TCommand : IParseCommand<TInput, TValue>
    {
        internal TCommand Command;

        public Parser(TCommand command)
        {
            this.Command = command;
        }

        public override ParseResult<TInput, TValue>? Parse(TInput input)
        {
            return this.Command.Execute(input);
        }

    }


    internal readonly struct ParserOrCommand<TInput, TValue> : IParseCommand<TInput, TValue>
    {
        private readonly Parser<TInput, TValue> _first;
        private readonly Parser<TInput, TValue> _second;

        public ParserOrCommand(Parser<TInput, TValue> first, Parser<TInput, TValue> second)
        {
            _first = first;
            _second = second;
        }

        public ParseResult<TInput, TValue>? Execute(TInput input)
        {
            return _first.Parse(input) ?? _second.Parse(input);
        }
    }

    internal readonly struct ParserInsertCommand<TInput, TValue> : IParseCommand<TInput, TValue>
    {
        readonly TValue _value;

        public ParserInsertCommand(TValue value)
        {
            this._value = value;
        }

        public ParseResult<TInput, TValue>? Execute(TInput input)
        {
            return new ParseResult<TInput, TValue>(this._value, input);
        }
    }
}