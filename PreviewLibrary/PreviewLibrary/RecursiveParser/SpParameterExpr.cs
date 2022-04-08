using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class SpParameterExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr Value { get; set; }
		public string OutToken { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{Name}");
			if (Value != null)
			{
				sb.Append($"={Value}");
			}
			if (!string.IsNullOrEmpty(OutToken))
			{
				sb.Append(" " + OutToken);
			}
			return sb.ToString();
		}
	}
}