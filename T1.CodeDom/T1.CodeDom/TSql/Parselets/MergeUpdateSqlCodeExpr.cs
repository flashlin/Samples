using System.Collections.Generic;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class MergeUpdateSqlCodeExpr : SqlCodeExpr
	{
		public List<AssignSqlCodeExpr> SetList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("UPDATE SET ");
			SetList.WriteToStreamWithCommaLine(stream);
		}
	}
}