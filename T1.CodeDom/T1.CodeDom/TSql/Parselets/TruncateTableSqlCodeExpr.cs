using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class TruncateTableSqlCodeExpr : SqlCodeExpr 
	{
		public SqlCodeExpr TableName { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("TRUNCATE TABLE ");
			TableName.WriteToStream(stream);
		}
	}
}