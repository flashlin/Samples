using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests
{
	public class GrantTest : TestBase
	{
		public GrantTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void grant_execute_on_object_to_role_as()
		{
			var sql = @"grant execute on object::[dbo].[myFunc] TO [RoleName] as [dbo]";

			Parse(sql);

			ThenExprShouldBe(@"GRANT EXECUTE ON OBJECT::[dbo].[myFunc] TO [RoleName] AS [dbo]");
		}
		
		[Fact]
		public void grant_alter()
		{
			var sql = @"GRANT ALTER
    ON OBJECT::[dbo].[Customer] TO [RoleName]
    AS [dbo];";
			Parse(sql);

			ThenExprShouldBe(@"GRANT ALTER ON OBJECT::[dbo].[Customer] TO [RoleName] AS [dbo] ;");
		}
	}
}
