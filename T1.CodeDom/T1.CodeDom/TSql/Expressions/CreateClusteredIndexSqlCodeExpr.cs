using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class CreateClusteredIndexSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr IndexName { get; set; }
		public SqlCodeExpr TableName { get; set; }
		public List<OrderItemSqlCodeExpr> OnColumns { get; set; }
		public SqlCodeExpr WithExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE CLUSTERED INDEX ");
			IndexName.WriteToStream(stream);
			stream.Write(" ON ");
			TableName.WriteToStream(stream);
			stream.Write("(");
			OnColumns.WriteToStreamWithComma(stream);
			stream.Write(")");

			if (WithExpr != null)
			{
				stream.Write(" ");
				WithExpr.WriteToStream(stream);
			}
		}
	}
	
	public class CreateIndexSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr IndexName { get; set; }
		public SqlCodeExpr TableName { get; set; }
		public List<SqlCodeExpr> OnColumns { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE INDEX ");
			IndexName.WriteToStream(stream);
			stream.Write(" ON ");
			TableName.WriteToStream(stream);
			stream.Write("(");
			OnColumns.WriteToStreamWithComma(stream);
			stream.Write(")");
		}
	}
}