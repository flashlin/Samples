using System.Collections;

namespace GitCli.Models.ConsoleMixedReality;

public static class HelpExtension
{
	public static string SubStr(this string str, int offset, int? length = null)
	{
		if (string.IsNullOrEmpty(str))
		{
			return string.Empty;
		}
		if (offset < 0)
		{
			return string.Empty;
		}
		if (offset >= str.Length)
		{
			return string.Empty;
		}

		if (length == null)
		{
			return str.Substring(offset);
		}

		var maxLen = str.Length - offset;
		var len = Math.Min(maxLen, length.Value);
		if( len == 0)
		{
			return string.Empty;
		}
		return str.Substring(offset, len);
	}


	public static string SubShowStr(this string str, int offset, int length)
	{
		if (string.IsNullOrEmpty(str))
		{
			return " ";
		}
		if (offset < 0)
		{
			offset = 0;
		}
		if (offset >= str.Length)
		{
			return " ";
		}
		var len = Math.Min(str.Length - offset, length);
		if (len <= 0)
		{
			return " ";
		}
		return str.Substring(offset, len);
	}

	public static IEnumerable<(T val, int idx)> WithIndex<T>(this IEnumerable<T> enumerable)
	{
		return enumerable.Select((val, idx) => (val, idx));
	}
}