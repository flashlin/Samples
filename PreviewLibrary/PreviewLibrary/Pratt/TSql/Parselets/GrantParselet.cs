using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Linq;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class GrantParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var permissionPrincipal = new[]
			{
				SqlToken.CONNECT,
			};

			var permissionList = parser.Scanner.ConsumeToStringListByDelimiter(SqlToken.Comma, permissionPrincipal)
				.ToList();

			parser.Scanner.Consume(SqlToken.To);

			var targetList = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
			{
				return parser.ParseExp() as SqlCodeExpr;
			}).ToList();

			return new GrantSqlCodeExpr
			{
				PermissionList = permissionList,
				TargetList = targetList
			};
		}
	}
}