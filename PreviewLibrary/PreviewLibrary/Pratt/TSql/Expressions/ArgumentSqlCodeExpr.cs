using PreviewLibrary.Pratt.Core;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class ArgumentSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public SqlCodeExpr DataType { get; set; }
		public SqlCodeExpr DefaultValueExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Name.WriteToStream(stream);
			stream.Write(" ");
			DataType.WriteToStream(stream);

			if (DefaultValueExpr != null)
			{
				stream.Write(" = ");
				DefaultValueExpr.WriteToStream(stream);
			}
		}
	}
}