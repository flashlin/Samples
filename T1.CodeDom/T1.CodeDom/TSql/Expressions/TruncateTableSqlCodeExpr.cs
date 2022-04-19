using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
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