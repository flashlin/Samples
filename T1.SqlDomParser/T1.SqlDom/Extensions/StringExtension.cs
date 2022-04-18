using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace T1.SqlDom.Extensions
{

	public static class StringExtension
	{
		public static string CondenseSpaces(this string s)
		{
			return s.Aggregate(new StringBuilder(), (acc, c) =>
			{
				if (c != ' ' || acc.Length == 0 || acc[acc.Length - 1] != ' ')
					acc.Append(c);
				return acc;
			}).ToString();
		}

		public static string TrimSpaces(this string s)
		{
			return s.Aggregate(new StringBuilder(), (acc, c) =>
			{
				if (c != ' ' || acc.Length == 0)
					acc.Append(c);
				return acc;
			}).ToString();
		}
	}
}
