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

			var joinSelectList = ParseJoinSelectList(parser);

			var unionSelectList = ParseUnionSelectList(parser);

			return new SelectSqlCodeExpr
			{
				Columns = columns,
				FromSourceList = fromSourceList,
				JoinSelectList = joinSelectList,
				WhereExpr = whereExpr,
				UnionSelectList = unionSelectList
			};
		}

		private List<SqlCodeExpr> ParseUnionSelectList(IParser parser)
		{
			var unionSelectList = new List<SqlCodeExpr>();
			do
			{
				if (!parser.Scanner.TryConsume(SqlToken.Union, out var unionSpan))
				{
					break;
				}
				var unionSelect = ParseUnionSelect(unionSpan, parser);
				unionSelectList.Add(unionSelect);
			} while (true);
			return unionSelectList;
		}

		private List<SqlCodeExpr> ParseJoinSelectList(IParser parser)
		{
			var joinSelectList = new List<SqlCodeExpr>();
			do
			{
				if (!parser.Scanner.TryConsumeAny(out var joinTypeSpan, SqlToken.Inner, SqlToken.Left, SqlToken.Right, SqlToken.Full, SqlToken.Cross))
				{
					break;
				}
				var joinSelect = ParseJoinSelect(joinTypeSpan, parser);
				joinSelectList.Add(joinSelect);
			} while (true);
			return joinSelectList;
		}

		private SqlCodeExpr ParseJoinSelect(TextSpan joinTypeSpan, IParser parser)
		{
			var parselet = new JoinParselet();
			return parselet.Parse(joinTypeSpan, parser) as SqlCodeExpr;
		}

		private static List<SqlCodeExpr> ParseFromSourceList(IParser parser)
		{
			var fromSourceList = new List<SqlCodeExpr>();
			if (parser.Scanner.TryConsume(SqlToken.From, out _))
			{
				fromSourceList = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
				{
					var sourceExpr = parser.ParseExp() as SqlCodeExpr;

					parser.TryConsumeAliasName(out var aliasNameExpr);

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

		protected SqlCodeExpr ParseUnionSelect(TextSpan unionToken, IParser parser)
		{
			var unionMethod = string.Empty;
			if (parser.Scanner.Match(SqlToken.All))
			{
				unionMethod = "ALL";
			}
			var rightExpr = parser.GetParseExpIgnoreCommentFunc()();
			return new UnionSelectSqlCodeExpr
			{
				UnionMethod = unionMethod,
				RightExpr = rightExpr,
			};
		}
	}
}
