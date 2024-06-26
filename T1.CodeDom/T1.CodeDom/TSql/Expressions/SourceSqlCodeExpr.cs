﻿using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class SourceSqlCodeExpr : SqlCodeExpr 
	{
		public SqlCodeExpr Column { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SOURCE.");
			Column.WriteToStream(stream);
		}
	}
}