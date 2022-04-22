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

		[Fact]
		public void update_set_comment()
		{
			var sql = @"update c
set name = @name, --test
id =@id
from customer c
";

			Parse(sql);

			ThenExprShouldBe(@"UPDATE c SET name = @name, id = @id
FROM customer AS c");
		}

		[Fact]
		public void update_set_var_eq()
		{
			var sql = @"UPDATE customer with (rowlock) SET @id=id=id+1";

			Parse(sql);

			ThenExprShouldBe(@"UPDATE customer WITH(rowlock) SET @id = id = id + 1");
		}


		[Fact]
		public void update_output_deleted()
		{
			var sql = @"update customer
		set usergroup = null
		output deleted.Id, deleted.username into @tmpCustomer (Id, username)
		where id = 1";

			Parse(sql);

			ThenExprShouldBe(@"UPDATE customer SET usergroup = NULL
OUTPUT deleted.Id, deleted.username
INTO @tmpCustomer(Id, username)
WHERE id = 1");
		}

		[Fact]
		public void update_into_date()
		{
			var sql = @"Update c
	set c.id = 4
	output inserted.id, inserted.birth, GETDATE()
	into customerLog(id,birth,date)
	from customer c, otherTable t
	where c.id = t.id";

			Parse(sql);

			ThenExprShouldBe(@"UPDATE c SET c.id = 4
OUTPUT inserted.id, inserted.birth, GETDATE()
INTO customerLog(id, birth, date)
FROM customer AS c, otherTable AS t
WHERE c.id = t.id");
		}

		[Fact]
		public void update_set_name_minus_eq()
		{
			var sql = @"update customer
	set id -= @id";

			Parse(sql);

			ThenExprShouldBe(@"UPDATE customer SET id -= @id");
		}

		[Fact]
		public void update_set_output_where()
		{
			var sql = @"update customer
		set 
			name = @name
		output
			1, deleted.Id into @customerTable
		where id = 1";

			Parse(sql);

			ThenExprShouldBe(@"UPDATE customer
SET name = @name
OUTPUT 1, deleted.Id 
INTO @customerTable
WHERE id = 1");
		}

	[Fact]
		public void update_set_output_aliasName()
		{
			var sql = @"update customer
		set 
			name = @name
		output
			1, deleted.Id sid into @customerTable
		where id = 1";

			Parse(sql);

			ThenExprShouldBe(@"UPDATE customer
SET name = @name
OUTPUT 1, deleted.Id AS sid
INTO @customerTable
WHERE id = 1");
		}

		
	}
}
