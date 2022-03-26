using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class ParenthesesTest : SqlTestBase
	{
		public ParenthesesTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void parentheses_arithmetic()
		{
			var sql = "(@a>=1 or @b + @c >=2)";

			var expr = _sqlParser.ParseParenthesesPartial(sql);

			"(@a >= 1 or @b + @c >= 2)".ShouldEqual(expr);
		}
	}
}