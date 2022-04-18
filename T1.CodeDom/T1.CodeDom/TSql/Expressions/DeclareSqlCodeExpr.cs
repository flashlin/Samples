using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class DeclareSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public SqlCodeExpr DataType { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("DECLARE ");
			Name.WriteToStream(stream);
			stream.Write(" ");
			DataType.WriteToStream(stream);
		}
	}
}