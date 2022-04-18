using System.Collections.Generic;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class ExprListSqlCodeExpr : SqlCodeExpr
	{
		public bool IsComma { get; set; } = true;

		public List<SqlCodeExpr> Items { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			if (IsComma)
			{
				Items.WriteToStreamWithComma(stream);
				return;
			}

			Items.WriteToStream(stream);
		}
	}
}