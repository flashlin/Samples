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
			var topCount = parser.ParseTopCount();

			var columns = new List<SqlCodeExpr>();
			do
			{
				columns.Add(ParseColumnAs(parser));
			} while (parser.Match(SqlToken.Comma));

			SqlCodeExpr intoTable = null;
			if(parser.Scanner.Match(SqlToken.Into))
			{
				intoTable = parser.ConsumeAny(SqlToken.TempTable) as SqlCodeExpr;
			}

			var fromSourceList = GetFrom_SourceList(parser);

			SqlCodeExpr pivotExpr = null;
			if (parser.Scanner.TryConsume(SqlToken.Pivot, out var pivotSpan))
			{
				pivotExpr = parser.PrefixParse(pivotSpan) as SqlCodeExpr;
			}

			SqlCodeExpr whereExpr = null;
			if (parser.Scanner.TryConsume(SqlToken.Where, out _))
			{
				whereExpr = parser.ParseExpIgnoreComment();
			}

			var groupBy = ParseGroupBy(parser);
			var orderBy = ParseOrderBy(parser);

			var unionSelectList = ParseUnionSelectList(parser);

			return new SelectSqlCodeExpr
			{
				TopCount = topCount,
				Columns = columns,
				IntoTable = intoTable,
				FromSourceList = fromSourceList,
				PivotExpr = pivotExpr,
				WhereExpr = whereExpr,
				GroupByList = groupBy,
				OrderByList = orderBy,
				UnionSelectList = unionSelectList
			};
		}

		private List<OrderItemSqlCodeExpr> ParseOrderBy(IParser parser)
		{
			var orderByList = new List<OrderItemSqlCodeExpr>();

			if (!parser.Scanner.TryConsume(SqlToken.Order, out _))
			{
				return orderByList;
			}

			parser.Scanner.Consume(SqlToken.By);
			do
			{
				if(!parser.TryConsumeObjectId(out var name))
				{
					name = parser.ParseExpIgnoreComment();
				}

				var ascOrDesc = "ASC";
				parser.Scanner.TryConsumeAny(out var ascOrDescSpan, SqlToken.Asc, SqlToken.Desc);
				if (!ascOrDescSpan.IsEmpty)
				{
					ascOrDesc = parser.Scanner.GetSpanString(ascOrDescSpan);
				}
				orderByList.Add(new OrderItemSqlCodeExpr
				{
					Name = name,
					AscOrDesc = ascOrDesc,
				});
			} while (parser.Scanner.Match(SqlToken.Comma));

			return orderByList;
		}

		private List<SqlCodeExpr> ParseGroupBy(IParser parser)
		{
			var groupByList = new List<SqlCodeExpr>();

			if (!parser.Scanner.TryConsume(SqlToken.Group, out _))
			{
				return groupByList;
			}

			parser.Scanner.Consume(SqlToken.By);
			do
			{
				var name = parser.ParseExpIgnoreComment();
				groupByList.Add(name);
			} while (parser.Scanner.Match(SqlToken.Comma));

			return groupByList;
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

		private static List<SqlCodeExpr> GetFrom_SourceList(IParser parser)
		{
			if (!parser.Scanner.TryConsume(SqlToken.From, out _))
			{
				return new List<SqlCodeExpr>();
			}
			return parser.ParseFromSourceList();
		}

		protected SqlCodeExpr ParseColumnAs(IParser parser)
		{
			var name = parser.ParseExpIgnoreComment();

			var meetColumnAliasNameList = new []
			{
				SqlToken.SqlIdentifier, SqlToken.Identifier, SqlToken.QuoteString,
				SqlToken.Date
			};

			TextSpan aliasNameToken;
			if (parser.Scanner.TryConsumeAny(out _, SqlToken.As))
			{
				aliasNameToken = parser.Scanner.ConsumeAny(meetColumnAliasNameList);
				return new ColumnSqlCodeExpr
				{
					Name = name,
					AliasName = parser.Scanner.GetSpanString(aliasNameToken),
				};
			}

			parser.Scanner.TryConsumeAny(out aliasNameToken, meetColumnAliasNameList);

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
			var rightExpr = parser.ParseExpIgnoreComment();
			return new UnionSelectSqlCodeExpr
			{
				UnionMethod = unionMethod,
				RightExpr = rightExpr,
			};
		}
	}
}
