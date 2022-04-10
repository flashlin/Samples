using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class OtherTest : TestBase
	{
		public OtherTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void multiComment()
		{
			var sql = @"/*
123
*/";
			Parse(sql);

			ThenExprShouldBe(@"/* 
123
*/");
		}

		[Fact]
		public void script_on_error_exit()
		{
			var sql = ":on error exit";
			
			Parse(sql);

			ThenExprShouldBe(":ON ERROR EXIT");
		}
	}
}
