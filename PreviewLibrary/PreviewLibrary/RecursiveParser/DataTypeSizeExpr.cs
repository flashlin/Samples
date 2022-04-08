using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class DataTypeSizeExpr : SqlExpr
	{
		public SqlExpr Size { get; set; }
		public int? ScaleSize { get; set; }

		public override string ToString()
		{
			var scale = "";
			if (ScaleSize != null)
			{
				scale = $",{ScaleSize}";
			}
			return $"({Size}{scale})";
		}
	}
}