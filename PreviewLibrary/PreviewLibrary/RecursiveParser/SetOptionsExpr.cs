using PreviewLibrary.Exceptions;
using System.Collections.Generic;

namespace PreviewLibrary.RecursiveParser
{
	public class SetOptionsExpr : SqlExpr
	{
		public List<string> Options { get; set; }
		public string Toggle { get; set; }

		public override string ToString()
		{
			var options = string.Join(",", Options);
			return $"SET {options} {Toggle}";
		}
	}
}