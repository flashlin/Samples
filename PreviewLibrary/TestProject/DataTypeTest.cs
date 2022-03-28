using Xunit;
using ExpectedObjects;
using Xunit.Abstractions;
using TestProject.Helpers;

namespace TestProject
{
	public class DataTypeTest : SqlTestBase
	{
		public DataTypeTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void table_primary_key()
		{
			var sql = "table ( id int primary key, rank int )";
			var expr = _sqlParser.ParseDataTypePartial(sql);

			@"TABLE (
id int PRIMARY KEY
,rank int
)".ShouldEqual(expr);
		}

		[Fact]
		public void max()
		{
			var sql = "nvarchar(max)";
			var expr = _sqlParser.ParseDataTypePartial(sql);

			"nvarchar(MAX)".ShouldEqual(expr);
		}

		[Fact]
		public void table_dataSize()
		{
			var sql = @"table
(
	BetOption nvarchar(10)
)";
			var expr = _sqlParser.ParseDataTypePartial(sql);

			@"TABLE (
BetOption nvarchar(10)
)".ShouldEqual(expr);
		}

		[Fact]
		public void tvpTable()
		{
			var sql = "tvpTable READONLY";
			var expr = _sqlParser.ParseDataTypePartial(sql);

			"tvpTable READONLY".ShouldEqual(expr);
		}
	}
}