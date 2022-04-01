using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary.Expressions
{
	public class ArgumentExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr DataType { get; set; }
		public SqlExpr DefaultValue { get; set; }
		public string OutputToken { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{Name} {DataType}");
			if(!string.IsNullOrEmpty(OutputToken))
			{
				sb.Append($" {OutputToken.ToUpper()}");
			}
			if (DefaultValue != null)
			{
				sb.Append($"={DefaultValue}");
			}
			return sb.ToString();
		}
	}
}