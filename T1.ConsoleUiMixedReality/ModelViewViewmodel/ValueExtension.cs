#nullable enable
namespace T1.ConsoleUiMixedReality.ModelViewViewmodel;

public static class ValueExtension
{
	public static bool IsEquals<T>(this T? value, T? other)
	{
		if (value != null)
		{
			return value.Equals(other);
		}
		if (other != null)
		{
			return other.Equals(value);
		}
		return true;
	}
}