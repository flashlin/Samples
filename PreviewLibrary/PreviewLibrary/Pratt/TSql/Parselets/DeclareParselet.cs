using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class DeclareParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var varName = parser.ConsumeAny(SqlToken.Variable) as SqlCodeExpr;
			var dataTypeExpr = parser.ConsumeDataType();
			return new DeclareSqlCodeExpr
			{
				Name = varName,
				DataType = dataTypeExpr,
			};
		}
	}
}