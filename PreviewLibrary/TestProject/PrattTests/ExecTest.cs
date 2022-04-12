using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class ExecTest : TestBase
	{
		public ExecTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void exec_func_a_b()
		{
			var sql = @"exec myFunc a,b";
			Parse(sql);

			ThenExprShouldBe(@"EXEC myFunc a, b ");
		}

		[Fact]
		public void exec_func_a_var_eq_b()
		{
			var sql = @"exec myFunc a, @b=b";
			Parse(sql);

			ThenExprShouldBe(@"EXEC myFunc a, @b = b");
		}

		[Fact]
		public void exec_func_var_eq_b()
		{
			var sql = @"exec myFunc @a='b'";
			Parse(sql);

			ThenExprShouldBe(@"EXEC myFunc @a = 'b'");
		}

	}
}
