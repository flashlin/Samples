using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class PrintSqlCodeExpr : SqlCodeExpr 
	{
		public SqlCodeExpr Value { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("PRINT ");
			Value.WriteToStream(stream);
		}
	}
}