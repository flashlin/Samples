using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class DataTypeSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr DataType { get; set; }
		public SqlCodeExpr SizeExpr { get; set; }
		public bool IsIdentity { get; set; }
		public bool IsReadOnly { get; set; }
		public SqlCodeExpr ConstraintExpr { get; set; }
		public bool IsPrimaryKey { get; set; }
		public bool IsNonclustered { get; set; }
		public SqlCodeExpr DefaultValueExpr { get; set; }
		public bool? IsAllowNull { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			DataType.WriteToStream(stream);

			if (IsIdentity)
			{
				stream.Write(" IDENTITY");
			}

			if (IsReadOnly)
			{
				stream.Write(" READONLY");
			}

			if (SizeExpr != null)
			{
				SizeExpr.WriteToStream(stream);
			}

			if (ConstraintExpr != null)
			{
				stream.Write(" ");
				ConstraintExpr.WriteToStream(stream);
			}

			if (IsPrimaryKey)
			{
				stream.Write(" PRIMARY KEY");
				if (IsNonclustered)
				{
					stream.Write(" NONCLUSTERED");
				}
			}

			if (DefaultValueExpr != null)
			{
				stream.Write(" ");
				DefaultValueExpr.WriteToStream(stream);
			}

			if (IsAllowNull != null)
			{
				if (IsAllowNull.Value)
				{
					stream.Write(" NULL");
				}
				else
				{
					stream.Write(" NOT NULL");
				}
			}
		}
	}
}