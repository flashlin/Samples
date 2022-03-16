using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class ArgumentExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr DataType { get; set; }
		public SqlExpr DefaultValue { get; set; }

		public override string ToString()
		{
			var defaultValue = string.Empty;
			if (DefaultValue != null)
			{
				defaultValue = $"={DefaultValue}";
			}
			return $"{Name} {DataType}{defaultValue}";
		}
	}
}