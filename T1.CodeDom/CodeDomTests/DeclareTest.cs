using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class DeclareTest : TestBase
	{
		public DeclareTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void declare_var_table()
		{
			var sql = @"declare @tb table(
id int,
name varchar(50)
)";
			Parse(sql);

			ThenExprShouldBe(@"DECLARE @tb TABLE
(
	id INT,
	name VARCHAR(50)
)");
		}

		[Fact]
		public void declare_var_eq_1()
		{
			var sql = @"declare @amount decimal (10,3)=1, @id int";

			Parse(sql);

			ThenExprShouldBe(@"DECLARE @amount DECIMAL(10,3) = 1
DECLARE @id INT");
		}

		[Fact]
		public void declare_cursor()
		{
			var sql = @"declare @p cursor";

			Parse(sql);

			ThenExprShouldBe(@"DECLARE @p CURSOR");
		}


		[Fact]
		public void declare_table_nonclustered()
		{
			var sql = @"declare @customer table(
id int primary key nonclustered
)";
			Parse(sql);

			ThenExprShouldBe(@"DECLARE @customer TABLE ( 
id INT PRIMARY KEY NONCLUSTERED
)");
		}
	}
}
