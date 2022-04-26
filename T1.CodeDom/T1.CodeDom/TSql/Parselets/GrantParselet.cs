using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
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
				SqlToken.Execute,
				SqlToken.Exec
			};

			var permissionList = parser.Scanner.ConsumeToStringListByDelimiter(SqlToken.Comma, permissionPrincipal)
				.ToList();

			if (permissionList.Count == 0)
			{
				var permissionPrincipalStr = string.Join(",", permissionPrincipal);
				ThrowHelper.ThrowParseException(parser, $"Expect one of {permissionPrincipalStr}.");
			}

			SqlCodeExpr onObjectId = null;
			if (parser.Scanner.Match(SqlToken.ON))
			{
				//onObjectId = parser.ConsumeAny(SqlToken.Object, SqlToken.SqlIdentifier, SqlToken.Identifier) as SqlCodeExpr;
				onObjectId = parser.ParseExpIgnoreComment();
			}

			parser.Scanner.Consume(SqlToken.To);

			var targetList = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
			{
				return parser.ParseExp() as SqlCodeExpr;
			}).ToList();


			SqlCodeExpr asDbo = null;
			if (parser.Scanner.Match(SqlToken.As))
			{
				asDbo = parser.ConsumeObjectId();
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