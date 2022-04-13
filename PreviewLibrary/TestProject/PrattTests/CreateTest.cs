using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class CreateTest : TestBase
	{
		public CreateTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void create_procedure()
		{
			var sql = @"create procedure myProc
@id int,
@name varchar(50)
as
begin
	set noexec on;
end";
			Parse(sql);

			ThenExprShouldBe(@"CREATE PROCEDURE myProc
@id INT, 
@name VARCHAR(50)
BEGIN
	SET NOEXEC ON
	; 
END");
		}

		[Fact]
		public void create_procedure_arg1_eq()
		{
			var sql = @"create procedure myProc
@id int,
@name varchar(50) = 'a'
as
begin
	set noexec on;
end";
			Parse(sql);

			ThenExprShouldBe(@"CREATE PROCEDURE myProc
@id INT, 
@name VARCHAR(50) = 'a'
BEGIN
	SET NOEXEC ON
	; 
END");
		}


	}
}
