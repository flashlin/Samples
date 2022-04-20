using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class FetchSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr CursorName { get; set; }
		public List<SqlCodeExpr> VariableNameList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("FETCH NEXT");

			stream.Write(" FROM ");
			CursorName.WriteToStream(stream);
			
			stream.WriteLine();
			stream.Write("INTO ");
			VariableNameList.WriteToStreamWithComma(stream);
		}
	}
}