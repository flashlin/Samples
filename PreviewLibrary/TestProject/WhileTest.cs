using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class WhileTest : SqlTestBase
	{
		public WhileTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void while_group()
		{
			var sql = @"while (@a = 1)
begin
	select 1
end
";
			var expr = _sqlParser.ParseWhilePartial(sql);
			@"WHILE @a = 1
BEGIN
SELECT 1
END".ShouldEqual(expr);
		}
	}

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