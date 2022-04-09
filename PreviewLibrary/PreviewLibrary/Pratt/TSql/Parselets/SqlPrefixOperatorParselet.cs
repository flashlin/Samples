using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;

namespace PreviewLibrary.Pratt.TSql.Parselets
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
