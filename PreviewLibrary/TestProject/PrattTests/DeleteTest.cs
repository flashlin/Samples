using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class DeleteTest : TestBase
	{
		public DeleteTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void delete()
		{
			var sql = @"delete customer WHERE id=@id";
			Parse(sql);
			ThenExprShouldBe(@"DELETE FROM customer WHERE id = @id");
		}
	}
}
