using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class InsertTest : TestBase
	{
		public InsertTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void insert_table_columns_values()
		{
			var sql = "insert customer (id,name) values(1,'a')";
			Parse(sql);
			ThenExprShouldBe(@"INSERT customer(id, name) VALUES 
(1, 'a')");
		}

		[Fact]
		public void insert_into_table_columns_values()
		{
			var sql = "insert into customer (id,name) values(1,'a')";
			Parse(sql);
			ThenExprShouldBe(@"INSERT INTO customer(id, name) VALUES 
(1, 'a')");
		}

		[Fact]
		public void insert_table_columns_values_negative()
		{
			var sql = "insert customer ([id], [name]) VALUES (-1, 'a')";
			Parse(sql);
			ThenExprShouldBe(@"INSERT customer([id], [name]) VALUES (-1, 'a')");
		}

		[Fact]
		public void insert_table_columns_values_1()
		{
			var sql = @"insert [dbo].[Countries] ([CountryID], [CountryName]) values (N';''', N';'' ', N'""; '':', NULL, -1)";
			Parse(sql);
			ThenExprShouldBe(@"INSERT [dbo].[Countries]([CountryID], [CountryName]) VALUES (N';''', N';'' ', N'""; '':', NULL, -1)");
		}


	}
}
