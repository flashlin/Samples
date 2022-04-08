using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class DeclareVariableExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr DataType { get; set; }
		public SqlExpr DefaultValue { get; set; }

		public override string ToString()
		{
			var defaultValue = "";
			if (DefaultValue != null)
			{
				defaultValue = " = " + DefaultValue.ToString();
			}
			return $"DECLARE {Name} {DataType}{defaultValue}";
		}
	}
}