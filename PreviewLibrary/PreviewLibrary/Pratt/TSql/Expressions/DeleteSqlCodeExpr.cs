using PreviewLibrary.Pratt.Core.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class DeleteSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Table { get; set; }
		public SqlCodeExpr WhereExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("DELETE FROM ");
			Table.WriteToStream(stream);
			if (WhereExpr != null)
			{
				stream.Write(" WHERE ");
				WhereExpr.WriteToStream(stream);
			}
		}
	}
}