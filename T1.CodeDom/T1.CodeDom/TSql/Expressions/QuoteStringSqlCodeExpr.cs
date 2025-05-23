﻿using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class QuoteStringSqlCodeExpr : SqlCodeExpr
	{
		public string Value { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write($"'{Value}'");
		}
	}
}