using System.Collections.Generic;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class IntoSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr IntoTable { get; set; }
		public List<SqlCodeExpr> ColumnList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("INTO ");
			IntoTable.WriteToStream(stream);
			stream.Write("(");
			ColumnList.WriteToStreamWithComma(stream);
			stream.Write(")");
		}
	}
}