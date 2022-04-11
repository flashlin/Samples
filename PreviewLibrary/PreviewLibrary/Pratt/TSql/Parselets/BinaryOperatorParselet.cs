using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class BinaryOperatorParselet : IInfixParselet
	{
		private readonly int _precedence;
		private readonly bool _isRight;

		public BinaryOperatorParselet(Precedence precedence, bool isRight)
		{
			this._precedence = (int)precedence;
			this._isRight = isRight;
		}

		public int GetPrecedence()
		{
			return _precedence;
		}

		public IExpression Parse(IExpression left, TextSpan token, IParser parser)
		{
			var oper = parser.Scanner.GetSpanString(token);
			var right = parser.ParseExp(_precedence - (_isRight ? 1 : 0));
			return new OperatorSqlCodeExpr
			{
				Left = left as SqlCodeExpr,
				Oper = oper,
				Right = right as SqlCodeExpr
			};
		}
	}
}