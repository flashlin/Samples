using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

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

			if (parser.TryConsumeToken(out var betweenSpan, SqlToken.Between))
			{
				var betweenExpr = (BetweenSqlCodeExpr)parser.ParseInfix(left as SqlCodeExpr, betweenSpan);
				return new NotBetweenSqlCodeExpr
				{
					Left = left	as SqlCodeExpr,
					Start = betweenExpr.StartExpr,
					End = betweenExpr.EndExpr
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

	public class NotBetweenSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			Left.WriteToStream(stream);
			stream.Write(" NOT BETWEEN ");
			Start.WriteToStream(stream);
			stream.Write(" AND ");
			End.WriteToStream(stream);
		}

		public SqlCodeExpr Left { get; set; }
		public SqlCodeExpr Start { get; set; }
		public SqlCodeExpr End { get; set; }
	}
}