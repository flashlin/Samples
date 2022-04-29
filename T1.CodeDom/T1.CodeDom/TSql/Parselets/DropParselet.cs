using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class DropParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if (parser.TryConsumeToken(out var roleSpan, SqlToken.ROLE))
			{
				return DropRole(roleSpan, parser);
			}
			
			if(parser.Scanner.IsToken(SqlToken.TABLE))
			{
				return DropTable(parser);
			}

			var helpMessage = parser.Scanner.GetHelpMessage();
			throw new ParseException(helpMessage);
		}

		private DropRoleSqlCodeExpr DropRole(TextSpan roleSpan, IParser parser)
		{
			var ifExists = parser.MatchTokenList(SqlToken.If, SqlToken.Exists);
			var name = parser.ConsumeObjectId();
			return new DropRoleSqlCodeExpr
			{
				IfExists = ifExists,
				RoleName = name,
			};
		}

		private DropSqlCodeExpr DropTable(IParser parser)
		{
			parser.Scanner.ConsumeString(SqlToken.TABLE);
			parser.TryConsume(SqlToken.TempTable, out var tmpTable);
			
			return new DropSqlCodeExpr
			{
				TargetId = "TABLE",
				ObjectId = tmpTable
			};
		}
	}

	public class DropRoleSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("DROP ROLE");
			if (IfExists)
			{
				stream.Write(" IF EXISTS");
			}

			stream.Write(" ");
			RoleName.WriteToStream(stream);
		}

		public bool IfExists { get; set; }
		public SqlCodeExpr RoleName { get; set; }
	}
}