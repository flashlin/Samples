using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class OtherTest : SqlTestBase
	{
		public OtherTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void waitfor_delay()
		{
			var sql = "waitfor delay '00:00:00.300'";
			var expr = _sqlParser.Parse(sql);
			"WAITFOR DELAY '00:00:00.300'".ShouldEqual(expr);
		}

		[Fact]
		public void begin_end()
		{
			var sql = @"begin select 1 end";
			var expr = _sqlParser.Parse(sql);
			"BEGIN SELECT 1 END".ShouldEqual(expr);
		}

		[Fact]
		public void commit()
		{
			var sql = @"commit";
			var expr = _sqlParser.Parse(sql);
			"COMMIT".ShouldEqual(expr);
		}

		[Fact]
		public void set_transaction_isolation_level_read_uncommitted()
		{
			var sql = @"set transaction isolation level read uncommitted";
			var expr = _sqlParser.Parse(sql);
			"SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED".ShouldEqual(expr);
		}

		[Fact]
		public void begin_tran()
		{
			var sql = @"begin tran";
			var expr = _sqlParser.Parse(sql);
			"BEGIN TRANSACTION".ShouldEqual(expr);
		}

		[Fact]
		public void rollback_transaction()
		{
			var sql = @"rollback transaction";
			var expr = _sqlParser.Parse(sql);
			"ROLLBACK TRANSACTION".ShouldEqual(expr);
		}


	}

}