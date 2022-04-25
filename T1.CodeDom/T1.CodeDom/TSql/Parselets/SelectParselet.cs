using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class SelectInsertContext
	{
		public TopSqlCodeExpr TopCount { get; set; }
		public List<SqlCodeExpr> Columns { get; set; }
		public SqlCodeExpr IntoTable { get; set; }
	}
	
	public class SelectParselet : IPrefixParselet
	{
		
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var topCount = parser.ParseTopCountExpr();

			var columns = parser.ParseColumnList();

			SqlCodeExpr intoTable = null;
			if (parser.Scanner.Match(SqlToken.Into))
			{
				intoTable = parser.ConsumeTableName();
			}

			var fromSourceList = GetFrom_SourceList(parser);

			SqlCodeExpr pivotExpr = null;
			if (parser.Scanner.TryConsumeAny(out var pivotSpan, SqlToken.Pivot, SqlToken.UnPivot))
			{
				pivotExpr = parser.PrefixParse(pivotSpan) as SqlCodeExpr;
			}

			SqlCodeExpr whereExpr = null;
			if (parser.Scanner.TryConsume(SqlToken.Where, out _))
			{
				whereExpr = parser.ParseExpIgnoreComment();
				whereExpr = parser.ParseLRParenExpr(whereExpr);
			}

			var groupBy = ParseGroupBy(parser);
			var having = ParseHaving(parser);
			var orderBy = ParseOrderBy(parser);
			
			SqlCodeExpr forXmlExpr = null;
			if (parser.TryConsumeToken(out var forSpan, SqlToken.For))
			{
				forXmlExpr = parser.PrefixParse(forSpan) as SqlCodeExpr;
			}

			var optionExpr = parser.ParseOptionExpr();

			var unionSelectList = ParseUnionSelectList(parser);

			var isSemicolon = parser.MatchToken(SqlToken.Semicolon);

			return new SelectSqlCodeExpr
			{
				TopCount = topCount,
				Columns = columns,
				IntoTable = intoTable,
				FromSourceList = fromSourceList,
				PivotExpr = pivotExpr,
				WhereExpr = whereExpr,
				GroupByList = groupBy,
				Having = having,
				OrderByList = orderBy,
				ForXmlExpr = forXmlExpr,
				OptionExpr = optionExpr,
				UnionSelectList = unionSelectList,
				IsSemicolon = isSemicolon,
			};
		}

		private IExpression ParseInsertExec(SelectInsertContext context, IParser parser)
		{
			var execExpr = parser.ParseExpIgnoreComment();
			return new InsertFromExecSqlCodeExpr
			{
				TopCount = context.TopCount,
				Columns = context.Columns,
				IntoTable = context.IntoTable,
				ExecExpr = execExpr,
			};
		}

		private HavingSqlCodeExpr ParseHaving(IParser parser)
		{
			if (!parser.MatchToken(SqlToken.Having))
			{
				return null;
			}
			var itemList = new List<SqlCodeExpr>();
			do
			{
				var item = parser.ParseExpIgnoreComment();
				itemList.Add(item);
			} while (parser.MatchToken(SqlToken.Comma));
			return new HavingSqlCodeExpr
			{
				ItemList = itemList
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
			orderByList = parser.ParseOrderItemList();
			/*
			do
			{
				var name = parser.ParseExpIgnoreComment();

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
			*/

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

	public class InsertFromExecSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("INSERT");

			if (TopCount != null)
			{
				stream.Write(" ");
				TopCount.WriteToStream(stream);
			}

			stream.Write(" ");
			IntoTable.WriteToStream(stream);

			if (Columns != null && Columns.Count > 0)
			{
				stream.Write(" ");
				Columns.WriteToStream(stream);
			}

			stream.Write(" EXEC ");
			ExecExpr.WriteToStream(stream);
		}

		public TopSqlCodeExpr TopCount { get; set; }
		public SqlCodeExpr IntoTable { get; set; }
		public List<SqlCodeExpr> Columns { get; set; }
		public SqlCodeExpr ExecExpr { get; set; }
	}

	public class HavingSqlCodeExpr : SqlCodeExpr
	{
		public List<SqlCodeExpr> ItemList { get; set; }
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("HAVING ");
			ItemList.WriteToStreamWithComma(stream);	
		}
	}
}
