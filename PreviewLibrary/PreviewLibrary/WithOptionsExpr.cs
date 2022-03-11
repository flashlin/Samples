using System.Collections.Generic;

namespace PreviewLibrary
{
	public class WithOptionsExpr
	{
		public List<string> Options { get; set; }

		public override string ToString()
		{
			if( Options == null)
			{
				return string.Empty;
			}
			return string.Join(",", Options);
		}
	}
}