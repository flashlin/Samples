using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace TestProject.PrattTests
{
	public static class SqlStringExtension
	{
		public static bool IsSql(this string text, string other)
		{
			return string.Equals(text, other, StringComparison.OrdinalIgnoreCase);
		}

		public static string MergeToCode(this string sql)
		{
			var sr = new StringReader(sql);
			var lines = new List<string>();
			do
			{
				var line = sr.ReadLine();
				if (line == null)
				{
					break;
				}
				lines.Add(line.Trim());
			} while (true);
			var singleLine = string.Join(" ", lines.Select(x => x));
			return singleLine;
		}

		public static List<string> ToLines(this string sql)
		{
			var sr = new StringReader(sql);
			var lines = new List<string>();
			do
			{
				var line = sr.ReadLine();
				if (line == null)
				{
					break;
				}
				lines.Add(line.Trim());
			} while (true);
			return lines;
		}

		public static string TrimCode(this string sql)
		{
			var sr = new StringReader(sql);
			var lines = new List<string>();
			do
			{
				var line = sr.ReadLine();
				if (line == null)
				{
					break;
				}
				lines.Add(line.Trim());
			} while (true);
			return string.Join("\r\n", lines);
		}
	}
}
