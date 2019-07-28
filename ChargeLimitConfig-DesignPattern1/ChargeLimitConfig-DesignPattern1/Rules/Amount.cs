using System.Collections.Generic;

namespace ChargeLimitConfig_DesignPattern1.Rules
{
	public class Amount<T>
			where T : struct
	{
		public static Amount<T> Unlimit = new Amount<T>(true, default(T));

		private readonly bool _isUnlimit;

		private readonly T _value;

		private Amount(bool isUnlimit, T value)
		{
			_isUnlimit = isUnlimit;
			_value = value;
		}

		public T Value => _value;

		public static int Comparison(Amount<T> a, Amount<T> b)
		{
			if (a._isUnlimit && b._isUnlimit)
			{
				return 0;
			}

			if (!a._isUnlimit && b._isUnlimit)
			{
				return -1;
			}

			if (a._isUnlimit && !b._isUnlimit)
			{
				return 1;
			}

			return Comparer<T>.Default.Compare(a._value, b._value);
		}

		public static Amount<T> From(T value)
		{
			var str = $"{value}";
			if (str == "0")
			{
				return Amount<T>.Unlimit;
			}

			return new Amount<T>(false, value);
		}

		public static bool operator !=(Amount<T> a, Amount<T> b)
		{
			return Comparison(a, b) != 0;
		}

		public static bool operator <(Amount<T> a, Amount<T> b)
		{
			return Comparison(a, b) < 0;
		}

		public static bool operator <=(Amount<T> a, Amount<T> b)
		{
			return Comparison(a, b) <= 0;
		}

		public static bool operator ==(Amount<T> a, Amount<T> b)
		{
			return Comparison(a, b) == 0;
		}

		public static bool operator >(Amount<T> a, Amount<T> b)
		{
			return Comparison(a, b) > 0;
		}

		public static bool operator >=(Amount<T> a, Amount<T> b)
		{
			return Comparison(a, b) >= 0;
		}

		public override bool Equals(object other)
		{
			if (other == null) return false;
			if (!(other is Amount<T>)) return false;
			return this == (Amount<T>)other;
		}

		public override int GetHashCode()
		{
			unchecked
			{
				return (_value.GetHashCode() * 397) ^ _isUnlimit.GetHashCode();
			}
		}

		protected bool Equals(Amount<T> other)
		{
			return _value.Equals(other._value) && _isUnlimit == other._isUnlimit;
		}
	}
}