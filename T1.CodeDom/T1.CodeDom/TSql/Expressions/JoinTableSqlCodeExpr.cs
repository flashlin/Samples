using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class JoinTableSqlCodeExpr : SqlCodeExpr
	{
		public string JoinType { get; set; }
		public string OuterType { get; set; }
		public SqlCodeExpr SecondTable { get; set; }
		public SqlCodeExpr AliasName { get; set; }
		public List<string> WithOptions { get; set; }
		public SqlCodeExpr JoinOnExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			if (!string.IsNullOrEmpty(JoinType))
			{
				stream.Write($"{JoinType.ToUpper()} ");
			}

			if (!string.IsNullOrEmpty(OuterType))
			{
				stream.Write($"{OuterType.ToUpper()} ");
			}
			stream.Write("JOIN ");

			SecondTable.WriteToStream(stream);
			if (AliasName != null)
			{
				stream.Write(" ");
				AliasName.WriteToStream(stream);
			}

			if (WithOptions != null && WithOptions.Count > 0)
			{
				stream.Write(" WITH(");
				WithOptions.WriteToStreamWithComma(stream);
				stream.Write(")");
			}

			if (JoinOnExpr != null)
			{
				stream.Write(" ");
				JoinOnExpr.WriteToStream(stream);
			}
		}
	}
}