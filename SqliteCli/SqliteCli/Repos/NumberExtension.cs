namespace SqliteCli.Repos
{
	public static class NumberExtension
	{
		public static string ToNumberString(this decimal number, int len)
		{
			return number.ToString("###,###,##0.00").ToFixLenString(len, AlignType.Right);
		}
	}
}
