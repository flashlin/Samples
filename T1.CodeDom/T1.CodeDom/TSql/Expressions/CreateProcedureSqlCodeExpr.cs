using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class CreateProcedureSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public List<ArgumentSqlCodeExpr> Arguments { get; set; }
		public SqlCodeExpr WithExecuteAs { get; set; }
		public List<SqlCodeExpr> Body { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE PROCEDURE ");
			Name.WriteToStream(stream);

			if (Arguments != null && Arguments.Count > 0)
			{
				stream.WriteLine();
				Arguments.WriteToStreamWithComma(stream);
			}

			if (WithExecuteAs != null)
			{
				stream.WriteLine();
				WithExecuteAs.WriteToStream(stream);
			}

			stream.WriteLine();
			stream.WriteLine("AS");
			Body.WriteToStream(stream);
		}
	}
}