using PreviewLibrary.Pratt.TSql.Expressions;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class DeleteParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var topCount = parser.ParseTopCountExpr();

			parser.Scanner.Match(SqlToken.From);

			var table = parser.ConsumeObjectId();

			var withOptions = parser.ParseWithOptions();

			var outputList = parser.GetOutputListExpr();
			var outputInto = parser.GetOutputIntoExpr();

			SqlCodeExpr whereExpr = null;
			if (parser.Scanner.Match(SqlToken.Where))
			{
				whereExpr = parser.ParseExp() as SqlCodeExpr;
			}

			return new DeleteSqlCodeExpr
			{
				TopExpr = topCount,
				Table = table,
				WithOptions = withOptions,
				OutputList = outputList,
				OutputInto = outputInto,
				WhereExpr = whereExpr
			};
		}
	}
}