using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests
{
	public class DeleteTest : TestBase
	{
		public DeleteTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void delete_table()
		{
			var sql = @"delete customer WHERE id=@id";
			Parse(sql);
			ThenExprShouldBe(@"DELETE FROM customer WHERE id = @id");
		}

		[Fact]
		public void delete_from_with_rowlock()
		{
			var sql = @"delete from customer with(rowlock) WHERE id=@id";
			Parse(sql);
			ThenExprShouldBe(@"DELETE FROM customer WITH(rowlock)
WHERE id = @id");
		}

		[Fact]
		public void delete_output()
		{
			var sql = @"delete from customer
	output deleted.id, deleted.name, 'System Message'
	into TrackCustomer([Id],[Name],[Desc])
	WHERE id = @id";

			Parse(sql);

			ThenExprShouldBe(@"DELETE FROM customer
OUTPUT deleted.id, deleted.name, 'System Message'
INTO TrackCustomer ([Id], [Name], [Desc])
WHERE id = @id");
		}

		[Fact]
		public void delete_output_from()
		{
			var sql = @"delete from customer
	output deleted.id, deleted.name, 'System Message'
	into TrackCustomer([Id],[Name],[Desc])
	from otherTable ds
		inner join @tIds bs on ds.Id = bs.Id
	WHERE ds.name = @name";

			Parse(sql);

			ThenExprShouldBe(@"DELETE FROM customer
OUTPUT deleted.id, deleted.name, 'System Message'
INTO TrackCustomer ([Id], [Name], [Desc])
FROM otherTable AS ds
INNER JOIN @tIds bs ON ds.Id = bs.Id
WHERE ds.name = @name");
		}

		[Fact]
		public void delete_top()
		{
			var sql = @"delete top (@batch) from customer";

			Parse(sql);

			ThenExprShouldBe(@"DELETE TOP ( @batch ) FROM customer");
		}
		
		
		[Fact]
		public void delete_output_into_output()
		{
			var sql = @"delete from c
output deleted.id
into otherCustomer(id)
output deleted.id
from @tmp t, customer c
where t.id = 1
";

			Parse(sql);

			ThenExprShouldBe(@"DELETE FROM c OUTPUT deleted.id
INTO otherCustomer (id) OUTPUT deleted.id
FROM @tmp AS t, customer AS c
WHERE t.id = 1");
		}
		
		
	}
}
