using Superpower;
using Superpower.Model;

namespace T1.SqlDomParser
{
	public static class Extensions
	{
		public static TextSpan UntilEnd(this TextSpan current, TextSpan? next)
		{
			if (next == null)
				return current;

			int absolute1 = next.Value.Position.Absolute + next.Value.Length;
			int absolute2 = current.Position.Absolute;
			return current.First(absolute1 - absolute2);
		}

		public static ParsedValue<T> ToParsedValue<T>(this T result, TextSpan span)
		{
			return new ParsedValue<T>(result, span);
		}

		public static ParsedValue<T> ToParsedValue<T>(this T result, TextSpan start, TextSpan end)
		{
			return new ParsedValue<T>(result, start.UntilEnd(end));
		}
	}
}
