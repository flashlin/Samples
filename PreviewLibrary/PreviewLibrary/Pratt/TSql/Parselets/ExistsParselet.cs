using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ExistsParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.LParen);
			
			var arguments = new List<SqlCodeExpr>();
			var innerExpr = parser.ParseExp() as SqlCodeExpr;
			arguments.Add(innerExpr);
			
			parser.Scanner.ConsumeAny(SqlToken.RParen);
			return new ExistsSqlCodeExpr
			{
				Name = "EXISTS",
				Parameters = arguments
			};
		}
	}
}