using PreviewLibrary.Pratt.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
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