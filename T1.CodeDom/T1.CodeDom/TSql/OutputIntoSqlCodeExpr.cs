using System.Collections.Generic;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql
{
	public class OutputIntoSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr IntoTable { get; set; }
		public List<SqlCodeExpr> ColumnsList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("INTO ");
			IntoTable.WriteToStream(stream);
			if (ColumnsList != null && ColumnsList.Count > 0)
			{
				stream.Write(" (");
				ColumnsList.WriteToStreamWithComma(stream);
				stream.Write(")");
			}
		}
	}
}