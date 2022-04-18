using PreviewLibrary.Pratt.TSql;
using PreviewLibrary.Pratt.TSql.Expressions;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class PrefixOperatorParselet : IPrefixParselet
	{
		private readonly int _precedence;

		public PrefixOperatorParselet(int precedence)
		{
			_precedence = precedence;
		}

		public IExpression Parse(TextSpan token, IParser parser)
		{
			// To handle right-associative operators like "^", we allow a slightly
			// lower precedence when parsing the right-hand side. This will let a
			// parselet with the same precedence appear on the right, which will then
			// take *this* parselet's result as its left-hand argument.
			var right = parser.ParseExp(_precedence) as SqlCodeExpr;

			return new PrefixExpression
			{
				Token = parser.Scanner.GetSpanString(token),
				Right = right
			};
		}

		public int GetPrecedence()
		{
			return _precedence;
		}
	}
}
