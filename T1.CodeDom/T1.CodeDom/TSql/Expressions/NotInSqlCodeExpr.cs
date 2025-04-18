﻿using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class NotInSqlCodeExpr : SqlCodeExpr 
	{
		public SqlCodeExpr Left { get; set; }
		public SqlCodeExpr Right { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Left.WriteToStream(stream);
			stream.Write(" NOT IN (");
			Right.WriteToStream(stream);
			stream.Write(")");
		}
	}
}