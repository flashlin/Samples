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

		[Fact]
		public void insert__into_var_select()
		{
			var sql = @"insert into @tmp select name from customer";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO @tmp SELECT name FROM customer");
		}

		[Fact]
		public void insert_into_var_columns_values()
		{
			var sql = @"insert into @customer (id, name)
      values
        (
            (case when @id>=0 then 0 else abs(@id1) end), -- test1
            (case when @id>=2 then abs(@id2) else 0 end), -- test2
            @id3, @name3
        )";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO @customer(id, name) VALUES
(( CASE
WHEN @id >= 0
THEN 0
ELSE abs( @id1 )
END ), ( CASE
WHEN @id >= 2
THEN abs( @id2 )
ELSE 0
END ), @id3, @name3)");
		}


		[Fact]
		public void insert_into_var_values()
		{
			var sql = @"insert into @res  values ( @start, @i, @word)";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO @res VALUES
(@start, @i, @word)");
		}

		[Fact]
		public void insert_into_table_columns_select()
		{
			var sql = @"insert into customer([id],[name])
select id,name from other_customer";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO customer
SELECT id, name
FROM other_customer");
		}
	}
}
