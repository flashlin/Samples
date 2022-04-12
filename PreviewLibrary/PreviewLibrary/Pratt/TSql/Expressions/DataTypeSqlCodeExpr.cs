using PreviewLibrary.Pratt.Core;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class DataTypeSqlCodeExpr : SqlCodeExpr
	{
		public string DataType { get; set; }
		public int? Size { get; internal set; }
		public int? Scale { get; internal set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(DataType.ToUpper());
			if (Size != null)
			{
				stream.Write($"({Size}");
				if (Scale != null)
				{
					stream.Write($",{Scale}");
				}
				stream.Write(")");
			}
		}
	}
}