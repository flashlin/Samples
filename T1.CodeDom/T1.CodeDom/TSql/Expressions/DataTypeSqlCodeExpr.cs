using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class DataTypeSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr DataType { get; set; }
		public bool IsReadOnly { get; set; }
		public int? Size { get; internal set; }
		public int? Scale { get; internal set; }
		public bool IsPrimaryKey { get; set; }
		public bool IsAllowNull { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			DataType.WriteToStream(stream);

			if(IsReadOnly)
			{
				stream.Write(" READONLY");
			}

			if (Size != null)
			{
				stream.Write($"(");
				if (Size == int.MaxValue)
				{
					stream.Write($"MAX");
				}
				else
				{
					stream.Write($"{Size}");
				}

				if (Scale != null)
				{
					stream.Write($",{Scale}");
				}
				stream.Write(")");
			}

			if (IsPrimaryKey)
			{
				stream.Write(" PRIMARY KEY");
			}

			if(IsAllowNull)
			{
				stream.Write(" NULL");
			}
		}
	}
}