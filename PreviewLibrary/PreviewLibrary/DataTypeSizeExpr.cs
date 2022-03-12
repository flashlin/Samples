namespace PreviewLibrary
{
	public class DataTypeSizeExpr : SqlExpr
	{
		public int Size { get; set; }
		public int? ScaleSize { get; set; }

		public override string ToString()
		{
			var scale = "";
			if( ScaleSize != null)
			{
				scale = $",{ScaleSize}";
			}
			return $"({Size}{scale})";
		}
	}
}