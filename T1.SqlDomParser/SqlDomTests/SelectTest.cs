using SqlDomTests.Helpers;
using System.Linq;
using T1.SqlDomParser;
using Xunit;
using Xunit.Abstractions;

namespace SqlDomTests
{
	public class SelectTest : SqlTestBase
	{
		public SelectTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void select_integer()
		{
			var sql = "select 1";
			Parse(sql);
			ThenResultShouldBe("SELECT 1");
		}

		[Fact]
		public void select_field1()
		{
			var sql = "select id";
			Parse(sql);
			ThenResultShouldBe("SELECT id");
		}

	}
}