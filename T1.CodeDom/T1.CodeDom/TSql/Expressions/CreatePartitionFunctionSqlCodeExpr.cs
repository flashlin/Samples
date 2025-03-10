﻿using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class CreatePartitionFunctionSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public SqlCodeExpr DataType { get; set; }
		public string RangeType { get; set; }
		public List<SqlCodeExpr> BoundaryValueList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE PARTITION ");
			Name.WriteToStream(stream);

			stream.Write("(");
			DataType.WriteToStream(stream);
			stream.Write(")");

			stream.Write("AS RANGE");
			if (!string.IsNullOrEmpty(RangeType))
			{
				stream.Write($" {RangeType}");
			}

			stream.WriteLine();
			stream.Write("FOR VALUES");
			stream.Write("(");
			BoundaryValueList.WriteToStreamWithComma(stream);
			stream.Write(")");
		}
	}
}