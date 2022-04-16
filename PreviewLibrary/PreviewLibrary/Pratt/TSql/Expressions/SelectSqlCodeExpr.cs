using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class SelectSqlCodeExpr : SqlCodeExpr
	{
		public int? TopCount { get; set; }
		public List<SqlCodeExpr> Columns { get; set; }
		public List<SqlCodeExpr> FromSourceList { get; set; }
		public List<SqlCodeExpr> JoinSelectList { get; set; }
		public SqlCodeExpr WhereExpr { get; set; }
		public List<SqlCodeExpr> UnionSelectList { get; set; }
		public List<OrderItemSqlCodeExpr> OrderByList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SELECT ");

			if (TopCount != null)
			{
				stream.Write($"TOP {TopCount} ");
			}

			foreach (var column in Columns.Select((value, idx) => new { value, idx }))
			{
				if (column.idx != 0)
				{
					stream.Write(", ");
				}
				column.value.WriteToStream(stream);
			}

			if (FromSourceList.Count > 0)
			{
				stream.WriteLine();
				stream.Write("FROM ");
				stream.Indent++;
				for (int i = 0; i < FromSourceList.Count; i++)
				{
					if (i != 0)
					{
						stream.WriteLine(", ");
					}
					var fromSource = FromSourceList[i];
					fromSource.WriteToStream(stream);
				}
				stream.Indent--;
			}

			if (JoinSelectList != null && JoinSelectList.Count > 0)
			{
				stream.WriteLine();
				JoinSelectList.WriteToStream(stream);
			}

			if (WhereExpr != null)
			{
				stream.WriteLine();
				stream.Write("WHERE ");
				WhereExpr.WriteToStream(stream);
			}

			if (OrderByList.Count > 0)
			{
				stream.WriteLine();
				stream.Write("ORDER BY ");
				OrderByList.WriteToStreamWithComma(stream);
			}

			if (UnionSelectList != null && UnionSelectList.Count > 0)
			{
				stream.WriteLine();
				UnionSelectList.WriteToStream(stream);
			}
		}
	}
}