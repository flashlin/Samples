using T1.CodeDom.TSql.Parselets;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class RowNumberSqlCodeExpr :  SqlCodeExpr
	{
		public OverSqlCodeExpr Over { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("ROW_NUMBER() ");
			Over.WriteToStream(stream);
		}
	}
}