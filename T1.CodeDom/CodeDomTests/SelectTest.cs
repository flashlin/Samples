using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class SelectTest : TestBase
	{
		public SelectTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void select_number()
		{
			var sql = "select 1";
			Parse(sql);
			ThenExprShouldBe("SELECT 1");
		}

		[Fact]
		public void select_string_as_aliasName()
		{
			var sql = "select 'abc' as error";
			Parse(sql);
			ThenExprShouldBe("SELECT 'abc' AS error");
		}

		[Fact]
		public void select_number_comment_number_from_table()
		{
			var sql = @"select 1, --test
2 from customer";
			Parse(sql);

			ThenExprShouldBe(@"SELECT 1, 2 FROM customer");
		}

		[Fact]
		public void select_name()
		{
			var sql = "select name";
			Parse(sql);
			ThenExprShouldBe("SELECT name");
		}

		[Fact]
		public void select_date()
		{
			var sql = "select date from customer";
			Parse(sql);
			ThenExprShouldBe("SELECT date FROM customer");
		}

		[Fact]
		public void select_customFunc()
		{
			var sql = "select [dbo].[isMy](123)";
			Parse(sql);
			ThenExprShouldBe("SELECT [dbo].[isMy]( 123 )");
		}

		[Fact]
		public void select_field_customFunc()
		{
			var sql = "select name, [dbo].[isMy](123)";
			Parse(sql);
			ThenExprShouldBe("SELECT name, [dbo].[isMy]( 123 )");
		}


		[Fact]
		public void select_group_customFunc()
		{
			var sql = "select name, ([dbo].[isMy](123)) as a";
			Parse(sql);
			ThenExprShouldBe("SELECT name, ( [dbo].[isMy]( 123 ) ) AS a");
		}

		[Fact]
		public void select_where_eq_and_between()
		{
			var sql = @"select id from customer where 
id1=@id and
id2 between 1 and 38 ";
			Parse(sql);

			ThenExprShouldBe(@"SELECT id
FROM customer
WHERE id1 = @id AND id2 BETWEEN 1 AND 38");
		}

		[Fact]
		public void select_var_eq_var_from_table_where_column_in_var()
		{
			var sql = @"select @id = @id + 1	
from customer with(nolock)
where id in (@id1, @id2)";
			
			Parse(sql);

			ThenExprShouldBe(@"SELECT @id = @id + 1
FROM customer WITH( nolock )
WHERE id IN (@id1, @id2)");
		}

		[Fact]
		public void select_name_from_table_order_by()
		{
			var sql = "select name from customer order by id";
			Parse(sql);
			ThenExprShouldBe(@"SELECT name
FROM customer
ORDER BY id ASC");
		}

		[Fact]
		public void select_name_from_table_group_by()
		{
			var sql = "select name from customer group by id";
			Parse(sql);
			ThenExprShouldBe(@"SELECT name
FROM customer
GROUP BY id");
		}



		[Fact]
		public void select_top_1_name()
		{
			var sql = "select top 1 name";
			Parse(sql);
			ThenExprShouldBe("SELECT TOP 1 name");
		}

		[Fact]
		public void select_name_number()
		{
			var sql = "select name, 1";
			Parse(sql);
			ThenExprShouldBe("SELECT name, 1");
		}

		[Fact]
		public void select_name_without_as_name()
		{
			var sql = "select customerName name";
			Parse(sql);
			ThenExprShouldBe("SELECT customerName AS name");
		}

		[Fact]
		public void select_name_as_name()
		{
			var sql = "select customerName as name";
			Parse(sql);
			ThenExprShouldBe("SELECT customerName AS name");
		}

		[Fact]
		public void select_number_from_dbo_table()
		{
			var sql = @"select 1 from dbo.customer";
			Parse(sql);
			ThenExprShouldBe("SELECT 1 FROM dbo.customer");
		}

		[Fact]
		public void select_number_from_table_aliasName()
		{
			var sql = @"select 1 from dbo.customer tb1";
			Parse(sql);
			ThenExprShouldBe("SELECT 1 FROM dbo.customer AS tb1");
		}

		[Fact]
		public void select_number_from_dbo_table_where()
		{
			var sql = @"select 1 from dbo.customer where name=customFunc()";
			Parse(sql);
			ThenExprShouldBe("SELECT 1 FROM dbo.customer WHERE name = customFunc()");
		}

		[Fact]
		public void select_number_from_dbo_table_where_and()
		{
			var sql = @"select 1 from dbo.customer where a=1 and b=2";
			Parse(sql);
			ThenExprShouldBe("SELECT 1 FROM dbo.customer WHERE a = 1 AND b = 2");
		}

		[Fact]
		public void select_number_union_all_select_number()
		{
			var sql = @"select 1 
union all
select 2";
			Parse(sql);
			ThenExprShouldBe("SELECT 1 UNION ALL SELECT 2");
		}

		[Fact]
		public void select_star_into_tmp()
		{
			var sql = @"select * into #tmp from customer";
			
			Parse(sql);

			ThenExprShouldBe(@"SELECT *
INTO #tmp
FROM customer");
		}

	[Fact]
		public void select_from_select()
		{
			var sql = @"SELECT c.id, p.*
        FROM (
                SELECT *
                FROM otherTable
                WHERE id = @id
        ) AS p
        JOIN customer c WITH (NOLOCK) ON p.id = c.id
        ORDER BY c.name, c.id;";
			
			Parse(sql);

			ThenExprShouldBe(@"SELECT c.id, p.*
FROM ( SELECT *
FROM otherTable
WHERE id = @id ) AS p
JOIN customer c WITH(NOLOCK) p.id = c.id
ORDER BY c.name ASC, c.id ASC");
		}
		

	[Fact]
		public void select_from_comment_select()
		{
			var sql = @"SELECT p.*
        FROM (
--test
                SELECT *
                FROM otherTable
                WHERE id = @id
        ) AS p
        ";
			
			Parse(sql);

			ThenExprShouldBe(@"SELECT p.*
FROM ( SELECT *
FROM otherTable
WHERE id = @id ) AS p");
		}


	[Fact]
		public void select_from_where_comment_comment()
		{
			var sql = @"select id
	from customer
	where c.id = 1 -- comment1
		and c.status & 2 <> 3 -- comment2
		and c.name is NULL
		";
			
			Parse(sql);

			ThenExprShouldBe(@"SELECT id
FROM customer
WHERE c.id = 1 AND c.status & 2 <> 3 AND c.name IS NULL");
		}


		[Fact]
		public void select_top_into_tmpTable_from_table()
		{
			var sql = @"SELECT TOP 1 * INTO #tmpCustomer from customer";
			
			Parse(sql);

			ThenExprShouldBe(@"SELECT TOP 1 *
INTO #tmpCustomer
FROM customer");
		}

		[Fact]
		public void select_with_index()
		{
			var sql = @"SELECT 1 from customer with(nolock, index(aaa))";
			
			Parse(sql);

			ThenExprShouldBe(@"SELECT 1
FROM customer WITH( nolock, INDEX(aaa) )");
		}

		[Fact]
		public void select_sum_as_out()
		{
			var sql = @"SELECT sum(id) as out, name from customer";
			
			Parse(sql);

			ThenExprShouldBe(@"SELECT SUM( id ) AS out, name FROM customer");
		}

		[Fact]
		public void select_name_as_count()
		{
			var sql = @"select 'main' as name, COUNT(1) Count";
			
			Parse(sql);

			ThenExprShouldBe(@"SELECT 'main' AS name, COUNT( 1 ) AS Count");
		}

	
		[Fact]
		public void select_having()
		{
			var sql = @"select 1 from customer where id=1 group by id having id > 1";
			
			Parse(sql);

			ThenExprShouldBe(@"SELECT 1 FROM customer
WHERE id = 1
GROUP BY id
HAVING id > 1");

		}
		

		[Fact]
		public void select_order_by()
		{
			var sql = @"select * from @customer c
	order by c.Status & 1, name";
			
			Parse(sql);

			ThenExprShouldBe(@"SELECT * FROM @customer AS c ORDER BY c.Status & 1 ASC, name ASC");

		}
		


	}
}
