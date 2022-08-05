namespace CodeSnippeter.Models;

public static class ValueExtension
{
	public static bool IsEquals<T>(this T value, T other)
	{
		if (value == null)
		{
			return other == null || other.Equals(value);
		}
		return value.Equals(other);
	}
}