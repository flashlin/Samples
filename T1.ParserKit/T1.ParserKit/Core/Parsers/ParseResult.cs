namespace T1.ParserKit.Core.Parsers
{
    public class ParseResult<TInput, TValue>
    {
        private TValue value;
        private TInput rest;

        public TValue Value { get { return this.value; } }
        public TInput Rest { get { return this.rest; } }

        public ParseResult(TValue value, TInput rest)
        {
            this.value = value;
            this.rest = rest;
        }

        public override string ToString()
        {
            return string.Format("Value: \"{0}\", Rest: \"{1}\"", this.value, this.rest);
        }
    }
}
