using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
	public class SqlPrefixOperatorParselet : IPrefixParselet
	{
		private PrefixOperatorParselet _prefix;

		public SqlPrefixOperatorParselet(Precedence precedence)
		{
			_prefix = new PrefixOperatorParselet((int)precedence);
		}

		public IExpression Parse(TextSpan token, IParser parser)
		{
			return _prefix.Parse(token, parser);
		}
	}
}
