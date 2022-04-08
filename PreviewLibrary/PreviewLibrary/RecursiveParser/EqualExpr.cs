using PreviewLibrary.Exceptions;
using System;

namespace PreviewLibrary.RecursiveParser
{
	public class EqualExpr : SqlExpr
	{
		public SqlExpr Left { get; set; }
		public SqlExpr Right { get; set; }

		public override string ToString()
		{
			return $"{Left} = {Right}";
		}
	}
}