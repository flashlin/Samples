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
				SqlToken.Delete,
				SqlToken.Insert,
				SqlToken.Select,
				SqlToken.Update,
				SqlToken.Execute
			};

			var permissionList = parser.Scanner.ConsumeToStringListByDelimiter(SqlToken.Comma, permissionPrincipal)
				.ToList();

			if (permissionList.Count == 0)
			{
				var permissionPrincipalStr = string.Join(",", permissionPrincipal);
				ThrowHelper.ThrowParseException(parser, $"Expect one of {permissionPrincipalStr}.");
			}

			SqlCodeExpr onObjectId = null;
			if (parser.Scanner.Match(SqlToken.On))
			{
				//onObjectId = parser.Scanner.ConsumeObjectId();
				onObjectId = parser.ConsumeAny(SqlToken.Object, SqlToken.SqlIdentifier, SqlToken.Identifier) as SqlCodeExpr;
			}

			parser.Scanner.Consume(SqlToken.To);

			var targetList = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
			{
				return parser.ParseExp() as SqlCodeExpr;
			}).ToList();


			SqlCodeExpr asDbo = null;
			if (parser.Scanner.Match(SqlToken.As))
			{
				asDbo = parser.Scanner.ConsumeObjectId();
			}

			return new GrantSqlCodeExpr
			{
				PermissionList = permissionList,
				OnObjectId = onObjectId,
				TargetList = targetList,
				AsDbo = asDbo
			};
		}
	}
}