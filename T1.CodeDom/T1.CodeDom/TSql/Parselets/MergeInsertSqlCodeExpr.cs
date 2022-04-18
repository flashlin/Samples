using System.Collections.Generic;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class MergeInsertSqlCodeExpr : SqlCodeExpr
	{
		public List<SqlCodeExpr> ColumnList { get; set; }
		public List<SqlCodeExpr> SourceColumnList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("INSERT ");
			stream.Write("(");
			ColumnList.WriteToStreamWithComma(stream);	
			stream.Write(") ");

			stream.Write("VALUES(");
			SourceColumnList.WriteToStreamWithComma(stream);
			stream.Write(")");
		}
	}
}