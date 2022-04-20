using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class OpenSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr CursorName { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("OPEN ");
			CursorName.WriteToStream(stream);
		}
	}
}