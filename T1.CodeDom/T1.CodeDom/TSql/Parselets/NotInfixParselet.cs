using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class NotInfixParselet : IInfixParselet
	{
		public IExpression Parse(IExpression left, TextSpan token, IParser parser)
		{
			if (parser.Scanner.TryConsumeAny(out var likeSpan, SqlToken.Like))
			{
				var right = parser.ParseExp();
				return new NotLikeSqlCodeExpr
				{
					Left = left as SqlCodeExpr,
					Right = right as SqlCodeExpr
				};
			}

			if (parser.Scanner.TryConsumeAny(out var inSpan, SqlToken.In))
			{
				var right = parser.ConsumeValueList();

				return new NotInSqlCodeExpr
				{
					Left = left as SqlCodeExpr,
					Right = right
				};
			}

			var helpMessage = parser.Scanner.GetHelpMessage(token);
			throw new ParseException(helpMessage);
		}

		public int GetPrecedence()
		{
			return (int)Precedence.CALL;
		}
	}
}