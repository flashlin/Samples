using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Linq;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ExecParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var funcName = parser.ConsumeAny(SqlToken.SqlIdentifier, SqlToken.Identifier) as SqlCodeExpr;

			var parameters = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
			{
				return parser.ParseExp() as SqlCodeExpr;
			}).ToList();

			return new ExecSqlCodeExpr
			{
				ExecToken = "EXEC",
				Name = funcName,
				Parameters = parameters
			};
		}
	}
}