using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class OutputSqlCodeExpr : SqlCodeExpr
	{
		public string OutputActionName { get; set; }
		public SqlCodeExpr ColumnName { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			if (!string.IsNullOrEmpty(OutputActionName))
			{
				stream.Write($"{OutputActionName}.");
			}
			ColumnName.WriteToStream(stream);
		}
	}
}