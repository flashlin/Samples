using Xunit;
using Xunit.Abstractions;

namespace TestProject.ParserTests
{
	public class CreateProcedureTest : ParserTestBase
	{
		public CreateProcedureTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void create_name_lparen_var_int_rrparen()
		{
			var sql = @"create procedure my_test
	@id int
as begin
	select 1
end
";
			Parse(sql);
			ThenExprShouldBe(@"CREATE PROCEDURE my_test
	@id INT
AS BEGIN
	SELECT 1
END");
		}
	}
}
