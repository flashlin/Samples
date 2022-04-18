using System.Collections.Immutable;

namespace T1.SqlDomParser
{
	public static class Operators
	{
		public enum Binary
		{
			Add,
			Sub,
			Mul,
			Div,
			And,
			Or,
			GreaterThan
		}

		public static ImmutableArray<ImmutableHashSet<Binary>> BinaryPrecedence { get; set; } = new[]
		{
			new[] { Binary.Mul, Binary.Div },
			new[] { Binary.Add, Binary.Sub },
			new[] { Binary.And },
			new[] { Binary.Or }
		}.Select(x => x.ToImmutableHashSet()).ToImmutableArray();

	}
}
