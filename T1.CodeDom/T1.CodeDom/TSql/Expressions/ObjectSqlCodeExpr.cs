﻿using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class ObjectSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Id { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("OBJECT::");
			Id.WriteToStream(stream);
		}
	}
}