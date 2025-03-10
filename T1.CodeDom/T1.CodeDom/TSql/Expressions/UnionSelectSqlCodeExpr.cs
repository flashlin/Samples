﻿using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class UnionSelectSqlCodeExpr : SqlCodeExpr
	{
		public string UnionMethod { get; set; }
		public SqlCodeExpr RightExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("UNION ");
			if (!string.IsNullOrEmpty(UnionMethod))
			{
				stream.Write($"{UnionMethod} ");
			}
			RightExpr.WriteToStream(stream);
		}
	}
}