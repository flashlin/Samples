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
AS
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
AS
BEGIN
	SET NOEXEC ON
	; 
END");
		}


		[Fact]
		public void create_table()
		{
			var sql = @"create table #cust (  
        ID int,
		birth datetime
    )";
			Parse(sql);

			ThenExprShouldBe(@"CREATE TABLE #cust(
ID INT,
birth DATETIME
)");
		}

		[Fact]
		public void create_table_comment()
		{
			var sql = @"create table #customer (       
id int,
   -- test --    
name varchar(10)
  )";
			Parse(sql);

			ThenExprShouldBe(@"CREATE TABLE #customer(
id INT,
name VARCHAR(10)
)");
		}

		[Fact]
		public void create_clustered_index()
		{
			var sql = @"create clustered index ix_id on #customer (id)";
			Parse(sql);

			ThenExprShouldBe(@"CREATE CLUSTERED INDEX ix_id ON #customer(id)");
		}

		[Fact]
		public void create_tmpTable_not_null()
		{
			var sql = @"create table #tmpCustomer
     (       
			id int NOT NULL
)";
			Parse(sql);

			ThenExprShouldBe(@"CREATE TABLE #tmpCustomer(
id INT NOT NULL
)");
		}
	}
}
