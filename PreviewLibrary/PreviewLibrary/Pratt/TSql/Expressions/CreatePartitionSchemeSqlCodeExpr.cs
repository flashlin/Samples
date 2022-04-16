using PreviewLibrary.Pratt.Core.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class CreatePartitionSchemeSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr SchemeName { get; set; }
		public SqlCodeExpr FuncName { get; set; }
		public string AllToken { get; set; }
		public List<SqlCodeExpr> GroupNameList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE PARTITION SCHEME ");
			SchemeName.WriteToStream(stream);
			stream.WriteLine();

			stream.Write("AS PARTITION ");
			FuncName.WriteToStream(stream);
			stream.WriteLine();

			if (!string.IsNullOrEmpty(AllToken))
			{
				stream.Write(AllToken);
			}

			stream.Write(" TO");
			stream.Write("(");
			GroupNameList.WriteToStreamWithComma(stream);
			stream.Write(")");
		}
	}
}