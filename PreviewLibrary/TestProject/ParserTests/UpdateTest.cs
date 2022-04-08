using Xunit;
using Xunit.Abstractions;

namespace TestProject.ParserTests
{
	public class UpdateTest : ParserTestBase
	{
		public UpdateTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void update_table_set_field_eq_var_where_field_eq_var()
		{
			var sql = @"update [customer]
set name=@name
where id=@id
";
			Parse(sql);
			ThenExprShouldBe(@"UPDATE [customer]
SET name=@name
WHERE id=@id");
		}
	}
}
