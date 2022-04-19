using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
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
			var right = parser.ParseExpIgnoreComment(_precedence - (_isRight ? 1 : 0));
			return new OperatorSqlCodeExpr
			{
				Left = left as SqlCodeExpr,
				Oper = oper.ToUpper(),
				Right = right as SqlCodeExpr
			};
		}
	}
}