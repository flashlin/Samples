using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class DeclareTest : TestBase
	{
		public DeclareTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void declare_var_table()
		{
			var sql = @"declare @tb table(
id int,
name varchar(50)
)";
			Parse(sql);

			ThenExprShouldBe(@"DECLARE @tb TABLE
(
	id INT,
	name VARCHAR(50)
)");
		}
	}
}
