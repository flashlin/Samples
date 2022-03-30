using Xunit;
using PreviewLibrary;
using FluentAssertions;
using System.Collections.Generic;
using FluentAssertions.Equivalency;
using ExpectedObjects;
using System.Linq;
using PreviewLibrary.Expressions;
using Xunit.Abstractions;
using TestProject.Helpers;

namespace TestProject
{
	public class SelectFromTest : SqlTestBase
	{
		public SelectFromTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void select_1_from_table_where_cast_add_cast()
		{
			var sql = "SELECT 1 FROM @a WHERE name = CAST( @b AS nvarchar(3) ) + ':' + CAST( @c AS nvarchar(3) )";
			var expr = _sqlParser.ParseSelectPartial(sql);
			sql.ShouldEqual(expr);
		}

		[Fact]
		public void select_column_from_table()
		{
			var sql = "select name from user";
			var expr = new SqlParser().Parse(sql);
			"SELECT name FROM user".ShouldEqual(expr);
		}

		[Fact]
		public void select_column_from_brackets_table()
		{
			var sql = "select name from [user]";
			var expr = new SqlParser().Parse(sql);
			"SELECT name FROM [user]".ShouldEqual(expr);
		}

		[Fact]
		public void select_column_aliasName_from_table()
		{
			var sql = "SELECT name n1 FROM user";
			
			var expr = _sqlParser.ParseSelectPartial(sql);

			"SELECT name as n1 FROM user".ShouldEqual(expr);
		}

		[Fact]
		public void select_column_as_aliasName_from_table()
		{
			var sql = "select name as n1 from user";
			var expr = new SqlParser().Parse(sql);
			"SELECT name as n1 FROM user".ShouldEqual(expr);
		}

		[Fact]
		public void select_column_as_aliasName_from_table_aliasName()
		{
			var sql = "SELECT name as n1 FROM user tb1";
			var expr = new SqlParser().Parse(sql);
			"SELECT name as n1 FROM user AS tb1".ShouldEqual(expr);
		}

		[Fact]
		public void select_column_as_aliasName_from_table_as_aliasName()
		{
			var sql = "SELECT name as n1 FROM user AS tb1";
			var expr = new SqlParser().Parse(sql);
			sql.ShouldEqual(expr);
		}

		[Fact]
		public void select_column_as_aliasName_from_table_as_aliasName_with_nolock()
		{
			var sql = "SELECT name as n1 FROM user AS tb1 WITH(nolock)";
			var expr = _sqlParser.Parse(sql);
			sql.ShouldEqual(expr);
		}

		[Fact]
		public void select_fields_from_table_nolock_where_like_and_greaterThan_or_smallerThan()
		{
			var sql = @"select CustID ,Transid, TransDate
	from Statement with (nolock)
	where TransDesc like 'Full Transfer%'
		and TransDate1 >= @from
		or TransDate2 < @to";

			var expr = _sqlParser.Parse(sql);

			@"SELECT CustID,Transid,TransDate
FROM Statement WITH(nolock)
WHERE TransDesc LIKE 'Full Transfer%'
and TransDate1 >= @from
or TransDate2 < @to".MergeToCode().ShouldEqual(expr);
		}

		[Fact]
		public void select_variable_eq_1_from_table()
		{
			var sql = "select @id = 1 from tb1";
			var expr = _sqlParser.ParseSelectPartial(sql);
			"SELECT @id = 1 FROM tb1".ShouldEqual(expr);
		}

		[Fact]
		public void select_func_from_table()
		{
			var sql = @"select CAST(@a as date) from customer";

			var expr = _sqlParser.ParseSelectPartial(sql);

			"SELECT CAST( @a AS date ) FROM customer".ShouldEqual(expr);
		}

		[Fact]
		public void Select2()
		{
			var sql = @"select 1
select 2";

			var exprs = new SqlParser().ParseAll(sql).ToList();

			@"SELECT 1
SELECT 2".ShouldEqual(exprs);
		}

		[Fact]
		public void select_variableName_eq_func1_from_tb1()
		{
			var sql = @"select @a = round((@b -1), 0) from tb1";
			
			var expr = _sqlParser.ParseSelectPartial(sql);

			"SELECT @a = round( (@b - 1),0 ) FROM tb1".ShouldEqual(expr);
		}

		[Fact]
		public void select_1_from_group_select_2()
		{
			var sql = "select 1 from (select 2)";
			var expr = _sqlParser.ParseSelectPartial(sql);

			"SELECT 1 FROM (SELECT 2)".ShouldEqual(expr);
		}

		[Fact]
		public void select_1_from_group_by()
		{
			var sql = "select 1 from tb1 group by id, name";
			var expr = _sqlParser.ParseSelectPartial(sql);
			@"SELECT 1 FROM tb1
	GROUP BY id,name".ShouldEqual(expr);
		}

		[Fact]
		public void select_1_from_table_where_name_eq_func0_and_func1_eq_string()
		{
			var sql = @"SELECT 1 FROM sys.databases WHERE name = DB_NAME() AND SUSER_SNAME( owner_sid ) = 'sa'";
			var expr = Parse(sql);
			sql.ShouldEqual(expr);
		}

		[Fact]
		public void select_1_from_table_where_name_eq_func0()
		{
			var sql = @"SELECT 1 FROM sys.databases WHERE name = DB_NAME()";
			var expr = Parse(sql);
			sql.ShouldEqual(expr);
		}

		[Fact]
		public void select_1_from_remote_table()
		{
			var sql = @"SELECT 1 FROM [remoteServer].[db].[dbo].[customer]";
			var expr = _sqlParser.ParseSelectPartial(sql);
			"SELECT 1 FROM [remoteServer].[db].[dbo].[customer]".ShouldEqual(expr);
		}

		[Fact]
		public void select_1_from_table_order_by_desc()
		{
			var sql = @"SELECT 1 FROM [customer] order by id desc";
			var expr = _sqlParser.ParseSelectPartial(sql);
			@"SELECT 1 FROM [customer]
ORDER BY id desc".ShouldEqual(expr);
		}

		[Fact]
		public void select_max_id_from_table()
		{
			var sql = @"select max(id) from customer";
			var expr = _sqlParser.ParseSelectPartial(sql);
			@"SELECT max( id ) FROM customer".ShouldEqual(expr);
		}

		[Fact]
		public void select_rank_over()
		{
			var sql = @"select field1, RANK() OVER (ORDER BY t.id DESC, t.price DESC) AS newRanking
				FROM @tb1, customer c with(nolock) where c.id = t.id
			";

			var expr = _sqlParser.ParseSelectPartial(sql);

			@"SELECT field1,RANK() OVER(
ORDER BY t.id DESC,t.price DESC
) AS newRanking FROM @tb1,customer AS c WITH(nolock) WHERE c.id = t.id".ShouldEqual(expr);
		}

		[Fact]
		public void select_from_join_from()
		{
			var sql = @"select f1 from tb1
				left join tb2 on tb1.id = tb2.id,
				tb3 as tb3 with(nolock),
				tb4
				where tb1.id = 123
			";

			var expr = _sqlParser.ParseSelectPartial(sql);

			@"SELECT f1 FROM tb1
Left JOIN tb2 ON tb1.id = tb2.id,tb3 AS tb3 WITH(nolock),tb4 WHERE tb1.id = 123".ShouldEqual(expr);
		}


	}
}
