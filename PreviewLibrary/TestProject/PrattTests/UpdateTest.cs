using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class UpdateTest : TestBase
	{
		public UpdateTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void update_set_field_eq_case()
		{
			var sql = @"update customer
set [name] = case when @id = -1 then [name] else @name end,
[desc] = @desc";

			Parse(sql);

			ThenExprShouldBe(@"UPDATE customer 
SET [name] = CASE WHEN @id = -1 THEN [name] ELSE @name END,
[desc] = @desc");
		}
	}
}
