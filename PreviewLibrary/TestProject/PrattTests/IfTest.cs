using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class IfTest : TestBase
	{
		public IfTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void if_begin_end()
		{
			var sql = @"if 'a' not like 'b'
begin
	set noexec on;
end";
			Parse(sql);

			ThenExprShouldBe(@"IF 'a' NOT LIKE 'b'
BEGIN
	SET NOEXEC ON ;
END");
		}
	}
}
