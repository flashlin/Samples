using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class OtherTest : TestBase
	{
		public OtherTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void multiComment()
		{
			var sql = @"/*
123
*/";
			Parse(sql);

			ThenExprShouldBe(@"/* 
123
*/");
		}

		[Fact]
		public void script_on_error_exit()
		{
			var sql = ":on error exit";

			Parse(sql);

			ThenExprShouldBe(":ON ERROR EXIT");
		}

		[Fact]
		public void identifier_not_like_var()
		{
			var sql = "name not like @name";

			Parse(sql);

			ThenExprShouldBe("name NOT LIKE @name");
		}

		[Fact]
		public void grant_connect_to_user()
		{
			var sql = "grant connect to [user_Name]";

			Parse(sql);

			ThenExprShouldBe("GRANT CONNECT TO [user_Name]");
		}

		[Fact]
		public void begin_tran()
		{
			var sql = "begin tran";

			Parse(sql);

			ThenExprShouldBe("BEGIN TRANSACTION");
		}

		[Fact]
		public void pivot()
		{
			var sql = "pivot (max(id) for idType in( [4], [3], [2] ) ) piv";

			Parse(sql);

			ThenExprShouldBe("PIVOT(max( id ) FOR idType IN ([4], [3], [2])) AS piv");
		}


		[Fact]
		public void cursor_for()
		{
			var sql = @"cursor for 
		  select Id, name from customer with (nolock)";

			Parse(sql);

			ThenExprShouldBe(@"CURSOR FOR SELECT Id, name
FROM customer WITH( nolock )");
		}

		[Fact]
		public void open()
		{
			var sql = @"open @mydata";

			Parse(sql);

			ThenExprShouldBe(@"OPEN @mydata");
		}
	}
}
