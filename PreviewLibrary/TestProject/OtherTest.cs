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


	}

}