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

		[Fact]
		public void update_table_with_rowlock()
		{
			var sql = @"update customer with(rowlock)
set id = 1";

			Parse(sql);

			ThenExprShouldBe(@"UPDATE customer WITH(rowlock) SET id = 1");
		}

		[Fact]
		public void update_table_output()
		{
			var sql = @"update [dbo].[customer] with(rowlock, updlock)
    set name = @name
    output deleted.Name
    where id = @id
";

			Parse(sql);

			ThenExprShouldBe(@"UPDATE [dbo].[customer] WITH(rowlock, updlock) SET name = @name
OUTPUT deleted.Name
WHERE id = @id");
		}


		[Fact]
		public void update_from_2table()
		{
			var sql = @"update c
set name = @name
from customer c, otherTable
    where c.id = @id
";

			Parse(sql);

			ThenExprShouldBe(@"UPDATE c SET name = @name
FROM customer AS c, otherTable
WHERE c.id = @id");
		}
	}
}
