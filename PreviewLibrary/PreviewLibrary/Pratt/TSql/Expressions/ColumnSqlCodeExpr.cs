using PreviewLibrary.Pratt.Core;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class ColumnSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public string AliasName { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Name.WriteToStream(stream);
			if (!string.IsNullOrEmpty(AliasName))
			{
				stream.Write($" AS {AliasName}");
			}
		}
	}
}