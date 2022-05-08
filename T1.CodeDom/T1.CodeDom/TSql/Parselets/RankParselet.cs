using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class RankParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if (!parser.IsToken(SqlToken.LParen))
			{
				return new ObjectIdSqlCodeExpr
				{
					ObjectName = parser.Scanner.GetSpanString(token)
				};
			}

			parser.Scanner.Consume(SqlToken.LParen);
			parser.Scanner.Consume(SqlToken.RParen);

			var overExpr = parser.Consume(SqlToken.Over) as OverSqlCodeExpr;

			return new RankSqlCodeExpr
			{
				Over = overExpr,
			};
		}
	}
	public class DenseRankParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if (!parser.IsToken(SqlToken.LParen))
			{
				return new ObjectIdSqlCodeExpr
				{
					ObjectName = parser.Scanner.GetSpanString(token)
				};
			}

			parser.Scanner.Consume(SqlToken.LParen);
			parser.Scanner.Consume(SqlToken.RParen);

			var overExpr = parser.Consume(SqlToken.Over) as OverSqlCodeExpr;

			return new RankSqlCodeExpr
			{
				Over = overExpr,
			};
		}
	}


	public static class SqlParseOverExtension
	{
		/*
		public static OverSqlCodeExpr ParseOver(this IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Over);
			parser.Scanner.Consume(SqlToken.LParen);

			var partitionColumnList = new List<SqlCodeExpr>();
			if (parser.Scanner.Match(SqlToken.Partition))
			{
				parser.Scanner.Consume(SqlToken.By);
				do
				{
					partitionColumnList.Add(parser.ConsumeObjectId());
				} while (parser.Scanner.Match(SqlToken.Comma));
			}

			parser.Scanner.Consume(SqlToken.Order);
			parser.Scanner.Consume(SqlToken.By);
			var orderColumnList = new List<SortSqlCodeExpr>();
			do
			{
				var name = parser.ConsumeObjectId();
				parser.Scanner.TryConsumeAny(out var sortTokenSpan, SqlToken.Asc, SqlToken.Desc);
				var sortToken = parser.Scanner.GetSpanString(sortTokenSpan);
				orderColumnList.Add(new SortSqlCodeExpr
				{
					Name = name,
					SortToken = sortToken
				});
			} while (parser.Scanner.Match(SqlToken.Comma));

			parser.Scanner.Consume(SqlToken.RParen);

			return new OverSqlCodeExpr
			{
				PartitionBy = new PartitionBySqlCodeExpr
				{
					ColumnList = partitionColumnList
				},
				OrderBy = new OrderBySqlCodeExpr
				{
					ColumnList = orderColumnList
				}
			};
		}
	*/
	}

	public class OverSqlCodeExpr : SqlCodeExpr
	{
		public PartitionBySqlCodeExpr PartitionBy { get; set; }
		public OrderBySqlCodeExpr OrderBy { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("OVER(");
			if (PartitionBy != null)
			{
				stream.Write(" ");
				PartitionBy.WriteToStream(stream);
			}
			if (OrderBy != null)
			{
				stream.Write(" ");
				OrderBy.WriteToStream(stream);
			}
			stream.Write(" )");
		}
	}


	public class PartitionBySqlCodeExpr : SqlCodeExpr
	{
		public List<SqlCodeExpr> ColumnList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("PARTITION BY ");
			ColumnList.WriteToStreamWithComma(stream);
		}
	}

	public class OrderBySqlCodeExpr : SqlCodeExpr
	{
		public List<SortSqlCodeExpr> ColumnList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("ORDER BY ");
			ColumnList.WriteToStreamWithComma(stream);
		}
	}
}
