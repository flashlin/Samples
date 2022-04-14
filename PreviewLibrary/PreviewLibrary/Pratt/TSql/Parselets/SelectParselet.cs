using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class SelectParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var columns = new List<SqlCodeExpr>();
			do
			{
				columns.Add(ParseColumnAs(parser));
			} while (parser.Match(SqlToken.Comma));

			var fromSourceList = ParseFromSourceList(parser);

			SqlCodeExpr whereExpr = null;
			if (parser.Scanner.TryConsume(SqlToken.Where, out _))
			{
				whereExpr = parser.ParseExp() as SqlCodeExpr;
			}

			return new SelectSqlCodeExpr
			{
				Columns = columns,
				FromSourceList = fromSourceList,
				WhereExpr = whereExpr
			};
		}

		private static List<SqlCodeExpr> ParseFromSourceList(IParser parser)
		{
			var fromSourceList = new List<SqlCodeExpr>();
			if (parser.Scanner.TryConsume(SqlToken.From, out _))
			{
				fromSourceList = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
				{
					var sourceExpr = parser.ParseExp() as SqlCodeExpr;

					TryConsumeAliasName(parser, out var aliasNameExpr);

					var userWithOptions = new List<string>();
					if (parser.Scanner.Match(SqlToken.With))
					{
						parser.Scanner.Consume(SqlToken.LParen);
						var withOptions = new[]
						{
								SqlToken.NOLOCK
							};
						userWithOptions = parser.Scanner.ConsumeToStringListByDelimiter(SqlToken.Comma, withOptions)
							.ToList();
						parser.Scanner.Consume(SqlToken.RParen);
					}

					return new FromSourceSqlCodeExpr
					{
						Left = sourceExpr,
						AliasName = aliasNameExpr,
						Options = userWithOptions,
					} as SqlCodeExpr;
				}).ToList();
			}

			return fromSourceList;
		}

		private static bool TryConsumeAliasName(IParser parser, out SqlCodeExpr aliasNameExpr)
		{
			if (parser.Scanner.TryConsumeObjectId(out aliasNameExpr))
			{
				return true;
			}

			if( parser.Scanner.Match(SqlToken.As))
			{
				return parser.Scanner.TryConsumeObjectId(out aliasNameExpr);
			}

			aliasNameExpr = null;
			return false;
		}

		protected SqlCodeExpr ParseColumnAs(IParser parser)
		{
			var name = parser.ParseExp() as SqlCodeExpr;

			TextSpan aliasNameToken;
			if (parser.Scanner.TryConsumeAny(out _, SqlToken.As))
			{
				aliasNameToken = parser.Scanner.ConsumeAny(SqlToken.SqlIdentifier, SqlToken.Identifier);
				return new ColumnSqlCodeExpr
				{
					Name = name,
					AliasName = parser.Scanner.GetSpanString(aliasNameToken),
				};
			}

			parser.Scanner.TryConsumeAny(out aliasNameToken, SqlToken.SqlIdentifier, SqlToken.Identifier);

			return new ColumnSqlCodeExpr
			{
				Name = name,
				AliasName = parser.Scanner.GetSpanString(aliasNameToken),
			};
		}
	}
}
