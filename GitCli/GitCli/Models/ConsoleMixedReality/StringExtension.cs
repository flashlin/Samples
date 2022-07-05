namespace GitCli.Models.ConsoleMixedReality;

public static class StringExtension
{
	public static string SubStr(this string str, int offset)
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
		return str.Substring(offset);
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
}