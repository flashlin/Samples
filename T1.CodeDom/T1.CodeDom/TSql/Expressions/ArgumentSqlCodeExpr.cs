﻿using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class ArgumentSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public SqlCodeExpr DataType { get; set; }
		public bool IsOutput { get; set; }
		public SqlCodeExpr DefaultValueExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Name.WriteToStream(stream);
			stream.Write(" ");
			DataType.WriteToStream(stream);

			if( IsOutput )
			{
				stream.Write(" OUTPUT");
			}

			if (DefaultValueExpr != null)
			{
				stream.Write(" = ");
				DefaultValueExpr.WriteToStream(stream);
			}
		}
	}
}