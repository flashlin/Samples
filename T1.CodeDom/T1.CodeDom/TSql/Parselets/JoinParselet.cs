using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class JoinParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan joinTypetoken, IParser parser)
		{
			var outerType = string.Empty;
			var joinTypeStr = parser.Scanner.GetSpanString(joinTypetoken);
			switch (joinTypeStr.ToUpper())
			{
				case "LEFT":
				case "RIGHT":
				case "FULL":
					if (parser.Match(SqlToken.Outer))
					{
						outerType = "OUTER";
					}
					break;
			}

			parser.ConsumeToken(SqlToken.Join);

			//var secondTable = parser.ConsumeAny(SqlToken.Variable, SqlToken.Identifier, SqlToken.SqlIdentifier) as SqlCodeExpr;
			var secondTable = parser.ParseExpIgnoreComment();
			// SqlCodeExpr secondTable;
			// if (parser.MatchToken(SqlToken.LParen))
			// {
			// 	secondTable = parser.ParseExpIgnoreComment();
			// 	parser.ConsumeToken(SqlToken.RParen);
			// 	secondTable = new GroupSqlCodeExpr
			// 	{
			// 		InnerExpr = secondTable,
			// 	};
			// }
			// else
			// {
			// 	secondTable = parser.ParseExpIgnoreComment();
			// }

			parser.TryConsumeAliasName(out var aliasNameExpr);

			var userWithOptions = parser.ParseWithOption();

			SqlCodeExpr joinOnExpr = null;
			if (parser.Scanner.Match(SqlToken.ON))
			{
				joinOnExpr = parser.ParseExpIgnoreComment();
				joinOnExpr = parser.ParseLRParenExpr(joinOnExpr);
			}

			return new JoinTableSqlCodeExpr
			{
				JoinType = joinTypeStr,
				OuterType = outerType,
				SecondTable = secondTable,
				AliasName = aliasNameExpr,
				WithOptions = userWithOptions,
				JoinOnExpr = joinOnExpr,
			};
		}
	}
}