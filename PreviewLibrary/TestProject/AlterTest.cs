using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class AlterTest : SqlTestBase
	{
		public AlterTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void alter_database()
		{
			var sql = @"ALTER DATABASE [$(DatabaseName)] ADD FILEGROUP [xxx]";

			var expr = _sqlParser.ParseAlterPartial(sql);
			@"ALTER DATABASE [$(DatabaseName)]
ADD FILEGROUP [xxx]".ShouldEqual(expr);
		}
	}

}