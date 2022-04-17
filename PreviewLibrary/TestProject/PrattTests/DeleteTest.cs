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
		public void delete_top()
		{
			var sql = @"delete top (@batch) from customer";

			Parse(sql);

			ThenExprShouldBe(@"DELETE TOP ( @batch ) FROM customer");
		}
	}
}
