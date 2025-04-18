﻿using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests
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
			ThenExprShouldBe(@"INSERT customer([id], [name]) VALUES
(-1, 'a')");
		}

		[Fact]
		public void insert_table_columns_values_1()
		{
			var sql = @"insert [dbo].[Countries] ([CountryID], [CountryName]) values (N';''', N';'' ', N'""; '':', NULL, -1)";
			Parse(sql);
			ThenExprShouldBe(@"INSERT [dbo].[Countries]([CountryID], [CountryName]) VALUES
(N';''', N';'' ', N'""; '':', NULL, -1)");
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
ELSE ABS( @id1 )
END ), ( CASE
WHEN @id >= 2
THEN ABS( @id2 )
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

			ThenExprShouldBe(@"INSERT INTO customer([id], [name])
SELECT id, name
FROM other_customer");
		}

		[Fact]
		public void insert_into_table_columns_output_into_select()
		{
			var sql = @"insert into customer([id],[name])
output 'customer', inserted.id, GETDATE()
into trackCustomer(id,name)
select id,name from other_customer";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO customer([id], [name])
OUTPUT 'customer', inserted.id, GETDATE()
INTO trackCustomer (id, name)
SELECT id, name
FROM other_customer");
		}

		[Fact]
		public void insert_into_exec()
		{
			var sql = @"insert into @customer
exec sp_read_customer
";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO @customer
EXEC sp_read_customer");
		}

		[Fact]
		public void insert_into_lparen()
		{
			var sql = @"insert into @customer(id)
( select 1 from customer )
";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO @customer(id)
( SELECT 1
FROM customer )");
		}

		[Fact]
		public void insert_into_comment()
		{
			var sql = @"insert into #customer(
id, --- test
name
)
( select 1, name from customer )
";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO #customer(id, name)
( SELECT 1, name
FROM customer )");
		}



		[Fact]
		public void insert_into_select_from_customFunc()
		{
			var sql = @"insert into @customer
		select val from myfunc(@id, N'a')";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO @customer SELECT val FROM myfunc( @id, N'a' )");
		}

		[Fact]
		public void insert_into_select_rank()
		{
			var sql = @"insert into @customer(id, name) select rank, name from otherTable;";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO @customer(id, name) SELECT rank, name FROM otherTable ;");
		}


		[Fact]
		public void insert_into()
		{
			var sql = @"INSERT INTO #customer (id, name)
    			EXEC [my_func] @date";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO #customer(id, name) EXEC [my_func] @date");
		}
		
		
		[Fact]
		public void insert_into_with_updlock()
		{
			var sql = @"INSERT INTO #customer with(updlock)
    			EXEC [my_func] @date";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO #customer WITH(UPDLOCK) EXEC [my_func] @date");
		}
		
		[Fact]
		public void insert_into_default()
		{
			var sql = @"INSERT INTO #customer values(1, default)";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO #customer VALUES (1, DEFAULT)");
		}
		
		
		[Fact]
		public void insert_into_password()
		{
			var sql = @"INSERT INTO #customer(password) values(1)";
			
			Parse(sql);

			ThenExprShouldBe(@"INSERT INTO #customer(password) VALUES (1)");
		}
		
		
	}
}
