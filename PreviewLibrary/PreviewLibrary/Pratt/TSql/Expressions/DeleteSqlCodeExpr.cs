using PreviewLibrary.Pratt.Core.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class DeleteSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Table { get; set; }
		public List<string> WithOptions { get; set; }
		public List<SqlCodeExpr> OutputList { get; set; }
		public SqlCodeExpr WhereExpr { get; set; }
		public SqlCodeExpr OutputInto { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("DELETE FROM ");
			Table.WriteToStream(stream);

			if (WithOptions != null && WithOptions.Count > 0)
			{
				stream.Write(" WITH(");
				WithOptions.WriteToStreamWithComma(stream);
				stream.Write(")");
			}

			if (OutputList != null && OutputList.Count > 0)
			{
				stream.WriteLine();
				stream.Write("OUTPUT ");
				OutputList.WriteToStreamWithComma(stream);
			}

			if (OutputInto != null)
			{
				stream.WriteLine();
				OutputInto.WriteToStream(stream);
			}

			if (WhereExpr != null)
			{
				stream.WriteLine();
				stream.Write("WHERE ");
				WhereExpr.WriteToStream(stream);
			}
		}
	}
}