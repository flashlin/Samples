using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class SelectSqlCodeExpr : SqlCodeExpr
	{
		public TopSqlCodeExpr TopCount { get; set; }
		public List<SqlCodeExpr> Columns { get; set; }
		public SqlCodeExpr IntoTable { get; set; }
		public List<SqlCodeExpr> FromSourceList { get; set; }
		public SqlCodeExpr PivotExpr { get; set; }
		public SqlCodeExpr WhereExpr { get; set; }
		public List<SqlCodeExpr> UnionSelectList { get; set; }
		public List<SqlCodeExpr> GroupByList { get; set; }
		public List<OrderItemSqlCodeExpr> OrderByList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SELECT");

			if (TopCount != null)
			{
				stream.Write(" ");
				TopCount.WriteToStream(stream);
			}

			stream.Write(" ");
			foreach (var column in Columns.Select((value, idx) => new { value, idx }))
			{
				if (column.idx != 0)
				{
					stream.Write(", ");
				}
				column.value.WriteToStream(stream);
			}

			if (IntoTable != null)
			{
				stream.WriteLine();
				stream.Write("INTO ");
				IntoTable.WriteToStream(stream);
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

			if (PivotExpr != null)
			{
				stream.WriteLine();
				PivotExpr.WriteToStream(stream);
			}

			if (WhereExpr != null)
			{
				stream.WriteLine();
				stream.Write("WHERE ");
				WhereExpr.WriteToStream(stream);
			}

			if (GroupByList.Count > 0)
			{
				stream.WriteLine();
				stream.Write("GROUP BY ");
				GroupByList.WriteToStreamWithComma(stream);
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