using PreviewLibrary.Pratt.Core.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class UpdateSqlCodeExpr : SqlCodeExpr
	{
		public int? TopCount { get; set; }
		public SqlCodeExpr Table { get; set; }
		public List<string> WithOptions { get; set; }
		public List<SqlCodeExpr> SetColumnsList { get; set; }
		public SqlCodeExpr WhereExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("UPDATE ");

			if(TopCount != null)
			{
				stream.Write($"TOP {TopCount} ");
			}

			Table.WriteToStream(stream);

			if(WithOptions != null && WithOptions.Count>0)
			{
				stream.Write(" WITH(");
				WithOptions.WriteToStreamWithComma(stream);
				stream.Write(")");
			}

			stream.Write(" SET ");
			SetColumnsList.WriteToStreamWithComma(stream);
			if (WhereExpr != null)
			{
				stream.WriteLine();
				stream.Write("WHERE ");
				WhereExpr.WriteToStream(stream);
			}
		}
	}
}