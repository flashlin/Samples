using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class ColumnDefineSqlCodeExpr : SqlCodeExpr
	{
		public string Name { get; set; }
		public SqlCodeExpr DataType { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Name);
			stream.Write(" ");
			DataType.WriteToStream(stream);
		}
	}
}