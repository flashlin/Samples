﻿using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class FromSourceSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Left { get; set; }
		public SqlCodeExpr AliasName { get; set; }
		public List<string> Options { get; set; }
		public List<SqlCodeExpr> JoinList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Left.WriteToStream(stream);

			if( AliasName != null)
			{
				stream.Write(" AS ");
				AliasName.WriteToStream(stream);
			}

			if (Options.Count > 0)
			{
				stream.Write(" WITH(");
				Options.ToUpper().WriteToStreamWithComma(stream);
				stream.Write(")");
			}

			if( JoinList != null && JoinList.Count > 0)
			{
				stream.WriteLine();
				JoinList.WriteToStream(stream);
			}
		}
	}
}