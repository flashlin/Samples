using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary
{
	public class InvokeFunctionExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{Name}()");
			return sb.ToString();
		}
	}
}