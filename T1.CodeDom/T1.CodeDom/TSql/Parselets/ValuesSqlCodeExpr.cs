using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class ValuesSqlCodeExpr : SqlCodeExpr
	{
		public List<SqlCodeExpr> ValueList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("VALUES (");
			ValueList.WriteToStreamWithComma(stream);
			stream.Write(")");
		}
	}
}