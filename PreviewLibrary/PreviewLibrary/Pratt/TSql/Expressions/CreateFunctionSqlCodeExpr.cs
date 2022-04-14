using PreviewLibrary.Pratt.Core.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class CreateFunctionSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public List<ArgumentSqlCodeExpr> Arguments { get; set; }
		public VariableSqlCodeExpr ReturnVariable { get; set; }
		public SqlCodeExpr ReturnType { get; set; }
		public List<SqlCodeExpr> Body { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE FUNCTION ");
			Name.WriteToStream(stream);
			stream.Write("(");
			Arguments.WriteToStreamWithComma(stream);
			stream.WriteLine(")");
			stream.Write("RETURNS ");
			if( ReturnVariable != null)
			{
				ReturnVariable.WriteToStream(stream);
				stream.Write(" ");
			}
			ReturnType.WriteToStream(stream);
			stream.WriteLine();
			stream.WriteLine("AS BEGIN");
			Body.WriteToStream(stream);
			stream.WriteLine();
			stream.Write("END");
		}
	}
}