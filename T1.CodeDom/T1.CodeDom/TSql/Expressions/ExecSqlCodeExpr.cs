using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class ExecSqlCodeExpr : SqlCodeExpr
	{
		public string ExecToken { get; set; }
		public SqlCodeExpr ReturnVariable { get; set; }
		public SqlCodeExpr Name { get; set; }
		public List<SqlCodeExpr> Parameters { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write($"{ExecToken}");
			if (ReturnVariable != null)
			{
				stream.Write(" ");
				ReturnVariable.WriteToStream(stream);
				stream.Write(" =");
			}

			stream.Write(" ");
			Name.WriteToStream(stream);

			stream.Write(" ");
			Parameters.WriteToStreamWithComma(stream);
		}
	}
}