using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class CreateTableSqlCodeExpr : SqlCodeExpr
	{
		public bool IsSemicolon { get; set; }
		public TableDataTypeSqlCodeExpr TableExpr { get; set; }
		public OnSqlCodeExpr OnPrimary { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE ");
			TableExpr.WriteToStream(stream);

			if (OnPrimary != null)
			{
				stream.Write(" ");
				OnPrimary.WriteToStream(stream);
			}
			
			if (IsSemicolon)
			{
				stream.Write(" ;");
			}
		}
	}
}