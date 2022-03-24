using ExpectedObjects;
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
END".ToExpectedObject().ShouldEqual(expr.ToString());
		}
	}
	}