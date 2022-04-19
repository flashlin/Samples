using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class PivotSqlCodeExpr :  SqlCodeExpr
	{
		public SqlCodeExpr Aggregated { get; set; }
		public SqlCodeExpr Column { get; set; }
		public List<SqlCodeExpr> PivotedColumns { get; set; }
		public SqlCodeExpr AliasName { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("PIVOT");
			stream.Write("(");
			Aggregated.WriteToStream(stream);
			stream.Write(" FOR ");
			Column.WriteToStream(stream);
			stream.Write(" IN (");
			PivotedColumns.WriteToStreamWithComma(stream);
			stream.Write(")");
			stream.Write(") AS ");
			AliasName.WriteToStream(stream);
		}
	}
}