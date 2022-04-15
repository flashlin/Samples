using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class JoinParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var outerType = string.Empty;
			var joinTypeStr = parser.Scanner.GetSpanString(token);
			switch(joinTypeStr.ToUpper())
			{
				case "LEFT":
				case "RIGHT":
				case "FULL":
					if( parser.Scanner.Match(SqlToken.Outer) )
					{
						outerType = "OUTER";
					}
					break;
			}

			parser.Scanner.Consume(SqlToken.Join);

			var secondTable = parser.Scanner.ConsumeObjectId();

			SqlCodeExpr joinOnExpr = null;
			if(parser.Scanner.Match(SqlToken.On))
			{
				joinOnExpr = parser.ParseExpIgnoreComment();
			}

			return new JoinTableSqlCodeExpr
			{
				JoinType = joinTypeStr,
				OuterType = outerType,
				SecondTable = secondTable,
				JoinOnExpr = joinOnExpr,
			};
		}
	}
}