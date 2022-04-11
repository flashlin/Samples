using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class NumberParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var tokenStr = parser.Scanner.GetSpanString(token);
			return new NumberSqlCodeExpr
			{
				Value = tokenStr
			};
		}
	}
	
	// public class IfParselet : IPrefixParselet
	// {
	// 	public IExpression Parse(TextSpan token, IParser parser)
	// 	{
	// 		var conditionExpr = parser.ParseExp();
	// 		parser.Scanner.ConsumeTokenType(SqlToken.B)
	// 		return new IfSqlCodeExpr
	// 		{
	// 			Condition = conditionExpr,
	// 		};
	// 	}
	// }
}
