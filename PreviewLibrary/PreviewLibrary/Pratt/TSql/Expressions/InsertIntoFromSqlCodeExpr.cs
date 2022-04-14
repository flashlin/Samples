using PreviewLibrary.Pratt.Core.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class InsertIntoFromSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Table { get; set; }
		public SqlCodeExpr SelectFromExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("INSERT INTO ");
			Table.WriteToStream(stream);
			stream.WriteLine();
			SelectFromExpr.WriteToStream(stream);
		}
	}
}