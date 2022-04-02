using FluentAssertions;
using T1.SqlDomParser;

namespace SqlDomTests.Helpers
{
	public static class TestExtensions
	{
		public static void ShouldBe(this SqlExpr sqlExpr, string expected)
		{
			sqlExpr.Should().NotBeNull();
			sqlExpr.ToSqlCode().Should().Be(expected);
		}
	}
}