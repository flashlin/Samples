using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class DeleteSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr TopExpr { get; set; }
		public SqlCodeExpr Table { get; set; }
		public List<string> WithOptions { get; set; }
		public List<SqlCodeExpr> OutputList { get; set; }
		public List<SqlCodeExpr> FromSourceList { get; set; }
		public SqlCodeExpr WhereExpr { get; set; }
		public SqlCodeExpr OutputInto { get; set; }
		public OptionSqlCodeExpr OptionExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("DELETE");

			if (TopExpr != null)
			{
				stream.Write(" ");
				TopExpr.WriteToStream(stream);
			}

			stream.Write(" FROM ");
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

			if (FromSourceList != null && FromSourceList.Count > 0)
			{
				stream.WriteLine();
				stream.Write("FROM ");
				FromSourceList.WriteToStreamWithComma(stream);
			}

			if (WhereExpr != null)
			{
				stream.WriteLine();
				stream.Write("WHERE ");
				WhereExpr.WriteToStream(stream);
			}

			if (OptionExpr != null)
			{
				stream.WriteLine();
				OptionExpr.WriteToStream(stream);
			}
		}
	}
}