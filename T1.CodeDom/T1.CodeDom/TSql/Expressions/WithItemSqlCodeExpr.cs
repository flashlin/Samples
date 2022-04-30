using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class WithItemSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Table { get; set; }
		public List<SqlCodeExpr> Columns { get; set; }
		public SqlCodeExpr InnerExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Table.WriteToStream(stream);

			if (Columns.Count > 0)
			{
				stream.Write("(");
				Columns.WriteToStreamWithComma(stream);
				stream.Write(")");
			}

			stream.WriteLine();
			stream.WriteLine("AS (");
			stream.Indent++;
			InnerExpr.WriteToStream(stream);
			stream.Indent--;
			stream.WriteLine();
			stream.Write(")");
		}
	}

	public class WithTableSqlCodeExpr : SqlCodeExpr
	{
		public List<WithItemSqlCodeExpr> Items { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("WITH ");
			Items.WriteToStreamWithCommaLine(stream);
		}
	}
}