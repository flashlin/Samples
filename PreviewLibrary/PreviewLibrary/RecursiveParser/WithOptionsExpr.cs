﻿using System.Collections.Generic;

namespace PreviewLibrary.RecursiveParser
{
	public class WithOptionsExpr
	{
		public List<string> Options { get; set; }

		public override string ToString()
		{
			if (Options == null)
			{
				return string.Empty;
			}
			return "WITH(" + string.Join(",", Options) + ")";
		}
	}
}