using PreviewLibrary.Pratt.Core.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class NotSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Right { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("NOT ");
			Right.WriteToStream(stream);
		}
	}
}