using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class IdentTest : SqlTestBase
	{
		public IdentTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void name()
		{
			var sql = @"name";
			var expr = _sqlParser.ParseSqlIdentPartial(sql);

			@"name".ShouldEqual(expr);
		}

		[Fact]
		public void dbo_name()
		{
			var sql = @"dbo.name";
			var expr = _sqlParser.ParseSqlIdentPartial(sql);

			@"dbo.name".ShouldEqual(expr);
		}

		[Fact]
		public void database_dbo_name()
		{
			var sql = @"db.dbo.name";
			var expr = _sqlParser.ParseSqlIdentPartial(sql);

			@"db.dbo.name".ShouldEqual(expr);
		}

		[Fact]
		public void remote_database_dbo_name()
		{
			var sql = @"remote.db.dbo.name";
			var expr = _sqlParser.ParseSqlIdentPartial(sql);

			@"remote.db.dbo.name".ShouldEqual(expr);
		}
	}
}